"""
Sapiens2 (Meta, humans only) without ComfyUI: segmentation, normals and matting.

Uses the official sapiens2 code (vendor/sapiens2) and the same checkpoints the ComfyUI
Sapiens2-Easy pack downloaded (models/sapiens2/<task>/sapiens2_<size>_<task>.safetensors).
Pre/post-processing is the same as that pack, so the results match what ComfyUI gave.
"""

import importlib
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F

from .. import config
from .base import device, free_vram, hf_download

SIZES = ("0.4b", "0.8b", "1b", "5b")

# task -> (checkpoint folder / file tag, config path inside the repo)
_TASKS = {
    "seg": ("seg", "sapiens/dense/configs/seg/shutterstock_goliath/sapiens2_{size}_seg_shutterstock_goliath-1024x768.py"),
    "normal": ("normal", "sapiens/dense/configs/normal/metasim_render_people/sapiens2_{size}_normal_metasim_render_people-1024x768.py"),
    "matting": ("matting", "sapiens/dense/configs/matting/gss_p3m_metasim/sapiens2_{size}_matting_gss_p3m_metasim-1024x768.py"),
}

SEG_PARTS = ("Background", "Apparel", "Eyeglass", "Face_Neck", "Hair", "Left_Foot", "Left_Hand",
             "Left_Lower_Arm", "Left_Lower_Leg", "Left_Shoe", "Left_Sock", "Left_Upper_Arm",
             "Left_Upper_Leg", "Lower_Clothing", "Right_Foot", "Right_Hand", "Right_Lower_Arm",
             "Right_Lower_Leg", "Right_Shoe", "Right_Sock", "Right_Upper_Arm", "Right_Upper_Leg",
             "Torso", "Upper_Clothing", "Lower_Lip", "Upper_Lip", "Lower_Teeth", "Upper_Teeth", "Tongue")
# eyeglass, face/neck, hair, lips, teeth, tongue
HEAD_CLASSES = (2, 3, 4, 24, 25, 26, 27, 28)

SEG_PALETTE = np.array([
    [50, 50, 50], [255, 218, 0], [14, 204, 182], [128, 200, 255], [255, 0, 109], [189, 0, 204],
    [255, 0, 218], [0, 160, 204], [0, 255, 145], [204, 0, 131], [182, 0, 255], [255, 109, 0],
    [0, 255, 255], [72, 0, 255], [204, 131, 0], [255, 0, 0], [72, 255, 0], [189, 204, 0],
    [182, 255, 0], [102, 0, 204], [32, 72, 204], [0, 145, 255], [14, 204, 0], [0, 128, 72],
    [235, 205, 119], [115, 227, 112], [157, 113, 143], [132, 93, 50], [82, 21, 114]], np.uint8)


def _repo():
    candidates = [config.PATHS["sapiens2_repo"],
                  os.path.join(config.PATHS["comfyui_dir"], "custom_nodes", "ComfyUI-Sapiens2-Easy", "vendor", "sapiens2")]
    for c in candidates:
        if os.path.isdir(os.path.join(c, "sapiens")):
            if c not in sys.path:
                sys.path.insert(0, c)
            return c
    raise RuntimeError("Sapiens2 code not found (looked in %s). Run setup.bat, or clone "
                       "https://github.com/facebookresearch/sapiens2 into vendor/sapiens2." % candidates)


def checkpoint(task, size, log=print):
    tag, _ = _TASKS[task]
    name = "sapiens2_%s_%s.safetensors" % (size, tag)
    p = config.find_model(name, "sapiens2/" + tag)
    if p:
        return p
    return hf_download("facebook/sapiens2-%s-%s" % (tag, size), name,
                       config.default_download_dir("sapiens2/" + tag), log=log)


def _dtype(dev):
    if dev.type != "cuda":
        return torch.float32
    return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16


class Sapiens2(object):
    """One loaded Sapiens2 dense model. Weights stay on the CPU between calls (see `gpu()`)."""

    def __init__(self, task, size="1b", log=print):
        if task not in _TASKS:
            raise ValueError("unknown Sapiens2 task %r" % task)
        repo = _repo()
        self.task, self.size = task, size
        ckpt = checkpoint(task, size, log=log)
        cfg_path = os.path.join(repo, _TASKS[task][1].format(size=size))
        if not os.path.isfile(cfg_path):
            raise FileNotFoundError("Sapiens2 config missing: %s" % cfg_path)
        importlib.import_module("sapiens.dense.src.models.init_model")   # registers the classes
        from sapiens.engine.config import Config
        from sapiens.engine.datasets import Compose
        from sapiens.registry import MODELS
        from safetensors.torch import load_file

        self.device = device()
        self.dtype = _dtype(self.device)
        log("loading Sapiens2 %s %s (%s)" % (task, size, os.path.basename(ckpt)))
        cfg = Config.fromfile(cfg_path)
        cfg.model["backbone"].pop("init_cfg", None)
        prev = torch.get_default_dtype()
        try:
            if self.dtype != torch.float32:
                torch.set_default_dtype(self.dtype)
            model = MODELS.build(cfg.model)
        finally:
            torch.set_default_dtype(prev)
        sd = load_file(ckpt, device="cpu")
        bad = model.load_state_dict(sd, strict=False)
        del sd
        if bad.missing_keys:
            log("  missing keys: %d (first: %s)" % (len(bad.missing_keys), bad.missing_keys[:3]))
        model.data_preprocessor = MODELS.build(cfg.data_preprocessor)
        model.pipeline = Compose(cfg.test_pipeline)
        self.model = model.eval()
        self._on_gpu = False

    # ---- VRAM: the pipeline moves the model on before a stage and off after ----
    def to_gpu(self):
        if self.device.type == "cuda" and not self._on_gpu:
            self.model.to(self.device)
            self._on_gpu = True

    def to_cpu(self):
        if self._on_gpu:
            self.model.to("cpu")
            self._on_gpu = False
            free_vram()

    def _run(self, rgb_u8):
        data = self.model.pipeline(dict(img=np.ascontiguousarray(rgb_u8[:, :, ::-1])))   # BGR, like cv2
        data = self.model.data_preprocessor(data)
        x = data["inputs"]
        if self.dtype != torch.float32:
            x = x.to(dtype=self.dtype)
        with torch.inference_mode():
            out = self.model(x)
        return out, data

    @staticmethod
    def _unpad(t, data):
        pad = data["data_samples"]["meta"].get("padding_size", (0, 0, 0, 0))
        if isinstance(pad, torch.Tensor):
            pad = pad.detach().cpu().tolist()
        l, r, tp, b = (int(v) for v in pad)
        return t[:, :, tp:t.shape[2] - b, l:t.shape[3] - r]

    def segment(self, rgb_u8):
        """-> (H, W) uint8 class ids (0 = background)."""
        logits, _ = self._run(rgb_u8)
        # same as Sapiens2-Easy: the seg logits are resized without removing the padding
        logits = F.interpolate(logits.float(), size=rgb_u8.shape[:2], mode="bilinear", align_corners=False)
        return logits.argmax(1)[0].byte().cpu().numpy()

    def normals(self, rgb_u8):
        """-> (3, h, w) float32 torch, unit normals in the Sapiens2 convention, crop resolution."""
        n, data = self._run(rgb_u8)
        n = n.float()
        n = n / torch.norm(n, dim=1, keepdim=True).clamp(min=1e-8)
        n = self._unpad(n, data)
        n = F.interpolate(n, size=rgb_u8.shape[:2], mode="bilinear", align_corners=False)[0]
        return n.cpu()

    def matte(self, rgb_u8):
        """-> (h, w) float32 alpha 0..1 (the model also predicts premult fg rgb, unused here)."""
        out, _ = self._run(rgb_u8)
        out = F.interpolate(out.float(), size=rgb_u8.shape[:2], mode="bilinear", align_corners=False)[0]
        return out[3].clamp(0, 1).cpu().numpy()


def colorize_seg(rgb_u8, class_ids, opacity=0.5):
    col = SEG_PALETTE[np.clip(class_ids, 0, len(SEG_PALETTE) - 1)]
    return (rgb_u8 * (1 - opacity) + col * opacity).astype(np.uint8)
