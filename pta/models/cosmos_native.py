"""
Cosmos DiffusionRenderer (NVIDIA inverse renderer) natively on Windows, no WSL.

Uses Materia (Biohazard VFX, GPL-3, github.com/NicholaiVogel/comfyui-materia): a pure-PyTorch
re-implementation of the Cosmos DiT (no Transformer Engine) + the diffusers Cosmos VAE. Only its
model files are used (vendor/materia), not its ComfyUI nodes.

Weights (Models window of the app, or python -m pta.download cosmos):
  <models>/diffusion_models/Diffusion_Renderer_Inverse_Cosmos_7B/model.pt   (nvidia, ~28 GB)
  <models>/vae/Cosmos-1.0-Tokenizer-CV8x8x8/vae/                             (diffusers VAE)

Same contract as the WSL runner: frames already letterboxed to 1280x704 in in_dir, 57-frame
chunks with a cross-faded overlap, one PNG per frame and pass in out_dir/<pass>/.
"""

import contextlib
import io
import os
import sys
import time
import types

import numpy as np
import torch

from .. import config
from .base import free_vram
from .cosmos import CosmosRenderer

INVERSE = "Diffusion_Renderer_Inverse_Cosmos_7B"
VAE = "Cosmos-1.0-Tokenizer-CV8x8x8"
SLOW = "slow, not stuck: 10-30 min per 57 frames, the bar moves once per chunk"
GBUFFER = {"basecolor": 0, "metallic": 1, "roughness": 2, "normal": 3, "depth": 4}


def materia_dir():
    d = os.path.join(str(config.APP_DIR), "vendor", "materia")
    if not os.path.isfile(os.path.join(d, "diffusion_renderer_pipeline.py")):
        raise RuntimeError("Materia code missing in vendor/materia: run setup.bat again.")
    return d


def _import_materia():
    """import Materia's model files as package 'materia' WITHOUT its __init__ (which loads the
    ComfyUI nodes, folder_paths and nvdiffrast)."""
    if "materia" not in sys.modules:
        pkg = types.ModuleType("materia")
        pkg.__path__ = [materia_dir()]
        sys.modules["materia"] = pkg
    from materia.CleanVAE import CleanVAE
    from materia.model_diffusion_renderer import CleanDiffusionRendererModel
    from materia.diffusion_renderer_pipeline import CleanDiffusionRendererPipeline
    from materia.diffusion_renderer_config import get_inverse_renderer_config
    return CleanVAE, CleanDiffusionRendererModel, CleanDiffusionRendererPipeline, get_inverse_renderer_config


def find_weights():
    """-> (model.pt path, VAE folder) or (None, None)."""
    ckpt = vae = None
    for root in config.PATHS["model_dirs"]:
        c = os.path.join(root, "diffusion_models", INVERSE, "model.pt")
        v = os.path.join(root, "vae", VAE, "vae")
        if ckpt is None and os.path.isfile(c):
            ckpt = c
        if vae is None and os.path.isfile(os.path.join(v, "config.json")):
            vae = v
    return ckpt, vae


def valid_frames(n):
    return 1 if n == 1 else ((n - 1 + 7) // 8) * 8 + 1


def chunks(n, size=57, overlap=8):
    """[(frame indices)] like the official split_list_with_overlap(chunk_mode='all')."""
    idx = list(range(n))
    step = max(1, size - overlap)
    out = []
    for i in range(0, max(1, n - overlap), step):
        out.append(idx[i:i + size])
    return out


class CosmosNative(CosmosRenderer):
    def __init__(self, s, log=print, say=None):
        super().__init__(s, log, say)
        self.pipe = None
        self._step, self._label = 0, ""

    def load(self):
        ckpt, vae_dir = find_weights()
        if ckpt is None or vae_dir is None:
            raise RuntimeError("Cosmos weights not found (%s / %s). Download them in the app: Models (bottom right)."
                               % (INVERSE, VAE))
        CleanVAE, Model, Pipeline, inverse_cfg = _import_materia()
        dtype = torch.bfloat16
        self.say("Loading Cosmos DiffusionRenderer (7B)")
        t0 = time.time()
        vae = CleanVAE(model_path=vae_dir)
        vae.reset_dtype(dtype)
        try:
            sd = torch.load(ckpt, map_location="cpu", weights_only=True, mmap=True)
        except Exception:
            sd = torch.load(ckpt, map_location="cpu", weights_only=False)
        if "model" in sd:
            sd = sd["model"]
        if "net.context_embedding.weight" not in sd:
            raise RuntimeError("%s is not the inverse renderer checkpoint" % ckpt)
        with torch.device("meta"):
            model = Model(inverse_cfg())
        model.to_empty(device="cpu")
        model.to(dtype=dtype)
        model.load_state_dict(sd, strict=True)
        del sd
        model.net.reinit_non_persistent_buffers()
        model.requires_grad_(False)
        model.train(False)
        self.pipe = Pipeline(checkpoint_dir=os.path.dirname(ckpt), checkpoint_name=os.path.basename(ckpt),
                             model_type=None, vae_instance=vae, model_instance=model,
                             guidance=float(self.s.cosmos_guidance), num_steps=int(self.s.cosmos_steps),
                             seed=1000, dtype=dtype)
        self.pipe.set_model_type("inverse")
        self._model = model
        self._hook_steps(model)
        self.log("[albedo] Cosmos loaded in %.0fs" % (time.time() - t0))

    def _hook_steps(self, model):
        """count the scheduler steps of the denoising loop -> 'Step 3 of 15' in the status line."""
        sch = getattr(model, "scheduler", None)
        if sch is None or getattr(sch, "_pta_hooked", False):
            return
        orig = sch.step
        me = self

        def step(*a, **k):
            me._step += 1
            me.say("%s  step %d of %d  (%s)" % (me._label, me._step, int(me.s.cosmos_steps), SLOW))
            return orig(*a, **k)

        sch.step = step
        sch._pta_hooked = True

    def run(self, in_dir, out_dir, progress=None, cancel=None):
        from PIL import Image
        files = sorted(f for f in os.listdir(in_dir) if f.endswith(".png"))
        n = len(files)
        passes = self.passes()
        if self.pipe is None:
            self.load()
        ch = chunks(n, 57, max(0, int(self.s.cosmos_overlap)))
        ov = max(0, int(self.s.cosmos_overlap))
        units, done = len(ch) * len(passes), 0
        acc = {p: {} for p in passes}
        self.pipe.to_gpu()
        try:
            for ci, idxs in enumerate(ch):
                nxt = ch[ci + 1][0] if ci + 1 < len(ch) else None
                T = valid_frames(len(idxs)) if len(ch) == 1 else 57
                pad = idxs + [idxs[-1]] * (T - len(idxs))
                vid = np.stack([np.asarray(Image.open(os.path.join(in_dir, files[i])).convert("RGB")) for i in pad])
                x = torch.from_numpy(vid).float().div(255.0).permute(3, 0, 1, 2)[None] * 2.0 - 1.0   # (1,C,T,H,W)
                for p in passes:
                    if cancel and cancel():
                        raise InterruptedError()
                    self._label = "Cosmos %s, frames %d-%d of %d" % (p, idxs[0] + 1, idxs[-1] + 1, n)
                    self._step = 0
                    self.say("%s  (%s)" % (self._label, SLOW))
                    batch = {"rgb": x, "video": x,
                             "context_index": torch.full((1, 1), GBUFFER[p], dtype=torch.long)}
                    m = getattr(self.pipe, "model", None)
                    if m is not None:
                        self._hook_steps(m)
                    noise = io.StringIO()              # Materia prints debug lines at every step
                    with torch.inference_mode(), contextlib.redirect_stdout(noise):
                        out = self.pipe.generate_video(data_batch=batch, normalize_normal=(p == "normal"),
                                                       seed=1000)[0]          # (T,H,W,3) uint8
                    self.say("%s  decoded" % self._label)
                    for k, fi in enumerate(idxs):
                        w = 1.0
                        if ci > 0 and k < ov:
                            w = min(w, (k + 1.0) / (ov + 1.0))
                        if nxt is not None and fi >= nxt:
                            w = min(w, (len(idxs) - k) / (ov + 1.0))
                        cur = acc[p].setdefault(fi, [np.zeros(out.shape[1:], np.float32), 0.0])
                        cur[0] += out[k].astype(np.float32) * w
                        cur[1] += w
                    os.makedirs(os.path.join(out_dir, p), exist_ok=True)
                    for fi in sorted(acc[p]):
                        if nxt is None or fi < nxt:
                            s, wsum = acc[p].pop(fi)
                            img = np.clip(s / max(wsum, 1e-6) + 0.5, 0, 255).astype(np.uint8)
                            Image.fromarray(img).save(os.path.join(out_dir, p, "%05d.png" % fi), compress_level=1)
                    done += 1
                    if progress:
                        progress(done, units)
        finally:
            self.pipe.to_cpu()
            free_vram()

    def stop(self):
        pass


def download(log=print):
    """Download the weights into the first model folder (asks for a Hugging Face token if needed)."""
    from huggingface_hub import snapshot_download, get_token, login
    if not get_token():
        print("A Hugging Face token with READ access is needed (huggingface.co > Settings > Access Tokens).")
        login()
    root = next((d for d in config.PATHS["model_dirs"] if os.path.isdir(d)), None)
    if root is None:
        root = config.default_download_dir("")
    log("downloading into %s" % root)
    snapshot_download(repo_id="nvidia/" + INVERSE, local_dir=os.path.join(root, "diffusion_models", INVERSE),
                      allow_patterns=["model.pt", "config.json", "README.md"])
    snapshot_download(repo_id="nvidia/" + VAE, local_dir=os.path.join(root, "vae", VAE),
                      allow_patterns=["vae/*", "README.md"])
    ckpt, vae = find_weights()
    log("model: %s\nvae:   %s" % (ckpt, vae))
    return ckpt, vae
