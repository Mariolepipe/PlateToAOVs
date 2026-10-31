"""Video Depth Anything (ByteDance), metric version: temporally stable depth in metres for the
whole sequence in one go (its own 32-frame windows), so there is no jump between chunks."""

import os
import sys

import cv2
import numpy as np
import torch

from .. import config
from .base import device, free_vram, hf_download

_CONFIGS = {
    "vits": {"encoder": "vits", "features": 64, "out_channels": [48, 96, 192, 384]},
    "vitb": {"encoder": "vitb", "features": 128, "out_channels": [96, 192, 384, 768]},
    "vitl": {"encoder": "vitl", "features": 256, "out_channels": [256, 512, 1024, 1024]},
}
_REPOS = {
    "video_depth_anything_vits.pth": "depth-anything/Video-Depth-Anything-Small",
    "video_depth_anything_vitb.pth": "depth-anything/Video-Depth-Anything-Base",
    "video_depth_anything_vitl.pth": "depth-anything/Video-Depth-Anything-Large",
    "metric_video_depth_anything_vits.pth": "depth-anything/Metric-Video-Depth-Anything-Small",
    "metric_video_depth_anything_vitb.pth": "depth-anything/Metric-Video-Depth-Anything-Base",
    "metric_video_depth_anything_vitl.pth": "depth-anything/Metric-Video-Depth-Anything-Large",
}


def _code():
    candidates = [config.PATHS["vda_code"],
                  os.path.join(config.PATHS["comfyui_dir"], "custom_nodes", "ComfyUI-Video-Depth-Anything")]
    for c in candidates:
        if os.path.isdir(os.path.join(c, "video_depth_anything")):
            if c not in sys.path:
                sys.path.insert(0, c)
            return c
    raise RuntimeError("Video Depth Anything code not found (looked in %s). Run setup.bat." % candidates)


class _Bar(object):
    def __init__(self, cb, total):
        self.cb, self.total, self.n = cb, total, 0

    def update(self, k):
        self.n = min(self.total, self.n + k)
        if self.cb:
            self.cb(self.n, self.total)


class VDA(object):
    def __init__(self, model_name, log=print):
        _code()
        from video_depth_anything.video_depth import VideoDepthAnything
        path = config.find_model(model_name, "videodepthanything")
        if path is None:
            path = hf_download(_REPOS[model_name], model_name, config.default_download_dir("videodepthanything"), log)
        enc = next(k for k in _CONFIGS if k in model_name)
        log("loading Video Depth Anything %s" % model_name)
        self.model = VideoDepthAnything(**_CONFIGS[enc], metric="metric" in model_name)
        try:
            sd = torch.load(path, map_location="cpu", weights_only=True)
        except Exception:
            sd = torch.load(path, map_location="cpu", weights_only=False)
        self.model.load_state_dict(sd, strict=True)
        self.model.eval()

    def infer(self, frames_u8, input_size=518, max_res=1280, fp16=True, progress=None):
        """frames_u8: (T, H, W, 3) uint8 -> (T, h, w) float32 depth (h, w = frames resized to max_res)."""
        T, H, W = frames_u8.shape[:3]
        if max_res > 0 and max(H, W) > max_res:          # same resize as the ComfyUI node
            s = max_res / float(max(H, W))
            h, w = round(H * s), round(W * s)
            h, w = h + h % 2, w + w % 2
            frames_u8 = np.stack([cv2.resize(f, (w, h), interpolation=cv2.INTER_AREA) for f in frames_u8])
        dev = device()
        self.model.to(dev)
        try:
            fn = self.model.infer_video_depth
            import inspect
            params = inspect.signature(fn).parameters
            if "pbar" in params:                          # ComfyUI node copy
                d = fn(frames_u8, input_size=input_size, device=dev.type, pbar=_Bar(progress, T), fp32=not fp16)
            else:                                         # upstream repo: (frames, target_fps, ...)
                d = fn(frames_u8, 24, input_size=input_size, device=dev.type, fp32=not fp16)
                d = d[0] if isinstance(d, tuple) else d
        finally:
            self.model.to("cpu")
            free_vram()
        return np.asarray(d, np.float32)[:T]
