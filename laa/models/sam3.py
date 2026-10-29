"""
SAM 3.1 (Meta, "Segment Anything 3", multiplex video tracker): finds every person in the shot from
the text "person" (Find everyone). Each one then starts SAM 2.1 (models/sam2.py), which follows them
with your keyframes.

ComfyUI's own SAM3 code is used as a library (same as MoGe / LTX): the SAM3 Detect / Video Track
node classes are called like functions. Weights: sam3.1_multiplex_fp16.safetensors in
models/checkpoints (Comfy-Org/sam3.1 on Hugging Face, no access request needed).

Results are boolean masks per object and per frame at the working size (see WORK_MAX).
"""

import numpy as np
import torch
import torch.nn.functional as F

from .. import config
from . import comfy_lib

MODEL = "sam3.1_multiplex_fp16.safetensors"
REPO = "Comfy-Org/sam3.1"
REPO_FILE = "checkpoints/sam3.1_multiplex_fp16.safetensors"
WORK_MAX = 1536            # frames are given to SAM3 at most this wide (it works at 1008 inside)


HUMAN_WORDS = {"person", "persons", "people", "human", "humans", "man", "men", "woman", "women", "child",
               "children", "kid", "kids", "boy", "girl", "actor", "actress", "soldier", "soldiers", "dancer"}


def prompts(text):
    """'person, car' -> ['person', 'car'] (SAM 3.1 looks for one thing at a time)"""
    out = [p.strip().lower() for p in (text or "").split(",") if p.strip()]
    return out or ["person"]


def is_human(prompt):
    return any(w in HUMAN_WORDS for w in prompt.replace("-", " ").split())


def work_size(W, H, max_side=WORK_MAX):
    s = min(1.0, float(max_side) / max(W, H))
    w, h = int(round(W * s)), int(round(H * s))
    return w + (w % 2), h + (h % 2)


class SAM3People(object):
    """say(text): one-line status for the UI."""

    def __init__(self, s, log=print, say=None):
        with torch.inference_mode():
            self._init(s, log, say)

    def _init(self, s, log, say):
        comfy_lib.setup(log)
        import nodes
        from comfy_extras import nodes_sam3
        self.nodes, self.ns = nodes, nodes_sam3
        self.s, self.log = s, log
        self.say = say or (lambda t: None)
        name = getattr(s, "sam3_model", MODEL) or MODEL
        config.require_model(name, "checkpoints", "Download it in the app: Models, bottom right (or turn off 'Number each "
                                                  "person' in Advanced settings).")
        self.say("Loading SAM 3.1")
        log("loading SAM3 %s" % name)
        out = nodes.CheckpointLoaderSimple().load_checkpoint(name)
        self.model, self.clip = comfy_lib.out(out, 0), comfy_lib.out(out, 1)
        self.cond = self.encode(prompts(getattr(s, "people_prompt", "person"))[0])

    def encode(self, prompt):
        """text prompt (English: person, car, house...) -> SAM3 conditioning"""
        with torch.inference_mode():
            return comfy_lib.out(self.nodes.CLIPTextEncode().encode(self.clip, prompt))

    # ------------------------------------------------------------------ calls
    def track(self, frames_u8, init_masks=None, detect=True, progress=None, cond=None):
        """
        frames_u8 : (N, h, w, 3) uint8, already at the working size
        init_masks: (O, h, w) bool masks on frames_u8[0] (objects to follow), or None
        detect    : also look for new people on every frame (text prompt)
        -> (N, O', h, w) bool. The first O objects are init_masks, in the same order.
        """
        with torch.inference_mode():
            N, h, w, _ = frames_u8.shape
            images = torch.from_numpy(np.ascontiguousarray(frames_u8)).float().div_(255.0)
            init = None
            if init_masks is not None and len(init_masks):
                init = torch.from_numpy(np.asarray(init_masks, np.float32))
            if init is None and not detect:
                return np.zeros((N, 0, h, w), bool)
            if progress:
                comfy_lib.set_progress(lambda v, t: progress(v, t))
            try:
                res = comfy_lib.out(self.ns.SAM3_VideoTrack.execute(
                    images=images, model=self.model, initial_mask=init,
                    conditioning=(self.cond if cond is None else cond) if detect else None,
                    detection_threshold=float(self.s.people_threshold),
                    max_objects=int(self.s.people_max), detect_interval=int(self.s.people_detect_interval)))
            finally:
                if progress:
                    comfy_lib.set_progress(None)
            return unpack(res, h, w)

    def release(self):
        comfy_lib.unload_all()


def unpack(track_data, h, w):
    """SAM3 track data (bit-packed masks at the model size) -> (N, O, h, w) bool."""
    from comfy.ldm.sam3.tracker import unpack_masks
    packed = track_data.get("packed_masks")
    N = track_data["n_frames"]
    if packed is None:
        return np.zeros((N, 0, h, w), bool)
    out = np.zeros((N, packed.shape[1], h, w), bool)
    for i in range(N):                                 # frame by frame: the full stack would be big
        m = unpack_masks(packed[i]).float()[None]      # (1, O, Hm, Wm)
        m = F.interpolate(m, size=(h, w), mode="bilinear", align_corners=False)[0]
        out[i] = (m > 0.5).cpu().numpy()
    return out


def download(log=print):
    from huggingface_hub import hf_hub_download
    import os
    import shutil
    root = next((d for d in config.PATHS["model_dirs"] if os.path.isdir(d)), None) or config.default_download_dir("")
    dst_dir = os.path.join(root, "checkpoints")
    os.makedirs(dst_dir, exist_ok=True)
    dst = os.path.join(dst_dir, MODEL)
    if os.path.isfile(dst):
        log("already there: %s" % dst)
        return dst
    log("downloading %s into %s" % (MODEL, dst_dir))
    p = hf_hub_download(repo_id=REPO, filename=REPO_FILE, local_dir=os.path.join(root, "_hf_sam3"))
    shutil.move(p, dst)
    shutil.rmtree(os.path.join(root, "_hf_sam3"), ignore_errors=True)
    log("done: %s" % dst)
    return dst
