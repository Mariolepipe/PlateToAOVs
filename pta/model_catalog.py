"""
Which model files each feature needs, where they come from, and whether they are already on this
PC (found by name in any model folder of config.json, e.g. your ComfyUI models). No torch import:
the app checks this at start-up and offers to download what is missing (pta.download does it).
"""

import os

from . import config

SAM2_CKPT = "sam2.1_hiera_large.pt"


def _sapiens(task, size):
    tag = {"seg": "seg", "normal": "normal"}[task]
    return "sapiens2_%s_%s.safetensors" % (size, tag)


def groups(s=None):
    """[{key, title, why, size, gated, licenses: [(name, url)], files: [names], needed: bool}]"""
    s = s or config.Settings()
    ltx_on = s.alpha_enabled and s.alpha_source == "ltx"
    return [
        dict(key="people", title="Segmentation: SAM 3.1 + SAM 2.1", why="SAM 3.1 finds what you type, SAM 2.1 follows your clicks",
             size="2.6 GB", gated=False, needed=bool(s.people_enabled),   # Comfy-Org/sam3.1: no login needed
             licenses=[("SAM 3.1", "https://huggingface.co/Comfy-Org/sam3.1"),
                       ("SAM 2.1", "https://github.com/facebookresearch/sam2")],
             files=[s.sam3_model, SAM2_CKPT]),
        dict(key="normals", title="Normals", why="Sapiens2 on the people, MoGe-3 on the set",
             size="7 GB", gated=False,     # depth uses MoGe's camera, Fast / Instant alpha use Sapiens
             needed=bool(s.normals_enabled or s.depth_enabled
                         or (s.alpha_enabled and s.alpha_source in ("matting", "sapiens"))),
             licenses=[("Sapiens2", "https://huggingface.co/facebook"), ("MoGe", "https://huggingface.co/Comfy-Org/MoGe")],
             files=[_sapiens("seg", s.seg_size), _sapiens("normal", s.sapiens_size), s.moge_model]),
        dict(key="depth", title="Depth and position", why="Video Depth Anything, metric",
             size="1.5 GB", gated=False, needed=bool(s.depth_enabled),
             licenses=[("Video Depth Anything", "https://huggingface.co/depth-anything/Metric-Video-Depth-Anything-Large")],
             files=[s.vda_model]),
        dict(key="matanyone", title="Alpha: Selection (MatAnyone 2)", why="the alpha of each number you selected",
             size="0.1 GB", gated=False, needed=bool(s.alpha_enabled and s.alpha_source == "matanyone"),
             licenses=[("MatAnyone 2 (non-commercial)", "https://github.com/pq-yang/MatAnyone2")],
             files=["matanyone2.pth"]),
        dict(key="alpha", title="Alpha: Foreground (LTX)", why="LTX-2.5 + the Alpha Gen IC-LoRA, whole frame",
             size="38 GB", gated=True, needed=bool(ltx_on),
             licenses=[("LTX-2.5", "https://huggingface.co/Lightricks/LTX-2.5"),
                       ("Alpha Gen IC-LoRA", "https://huggingface.co/Lightricks/LTX-2.5-22b-IC-LoRA-Alpha-Gen")],
             files=[s.ltx_unet, s.ltx_text_encoder, s.ltx_vae, s.ltx_lora]),
        dict(key="albedo", title="Albedo: best quality", why="Cosmos DiffusionRenderer (NVIDIA)",
             size="30 GB", gated=True,
             needed=bool(s.albedo_enabled and s.albedo_source == "cosmos" and s.cosmos_backend == "native"),
             licenses=[("Cosmos DiffusionRenderer", "https://huggingface.co/nvidia/Diffusion_Renderer_Inverse_Cosmos_7B")],
             files=["cosmos"]),
    ]


def _cosmos_ok():
    inv, vae = "Diffusion_Renderer_Inverse_Cosmos_7B", "Cosmos-1.0-Tokenizer-CV8x8x8"
    ok_c = ok_v = False
    for root in config.PATHS["model_dirs"]:
        ok_c = ok_c or os.path.isfile(os.path.join(root, "diffusion_models", inv, "model.pt"))
        ok_v = ok_v or os.path.isfile(os.path.join(root, "vae", vae, "vae", "config.json"))
    return ok_c and ok_v


def _index():
    """every file name in the model folders (one walk, instead of one per file)"""
    names = set()
    for root in config.PATHS["model_dirs"]:
        for dirpath, dirnames, filenames in os.walk(root):
            dirnames[:] = [d for d in dirnames if not d.startswith(".")]
            names.update(f for f in filenames if f.endswith((".safetensors", ".pt", ".pth", ".bin")))
    return names


def missing(group, index=None):
    """file names of this group that are not on this PC"""
    index = _index() if index is None else index
    out = []
    for f in group["files"]:
        if f == "cosmos":
            if not _cosmos_ok():
                out.append(f)
        elif f not in index:
            out.append(f)
    return out


def status(s=None):
    """{key: [missing files]} for every group"""
    index = _index()
    return {g["key"]: missing(g, index) for g in groups(s)}
