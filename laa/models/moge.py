"""MoGe-3 (monocular geometry): normals of the whole frame, depth, camera. Same call as the
ComfyUI MoGeInference node (fixed FOV, force_projection, no mask applied)."""

import numpy as np
import torch

from .. import config
from . import comfy_lib


MAX_Z_BINS = 4096          # depth bins of MoGe-3's sparse refiner (1/256 of log-depth each: 4096 = x8800 range)
STATS = {}


def _patch_refiner(log):
    """MoGe-3 bins log-depth to build the refiner's sparse 3D volume. A NaN/inf or an extreme depth
    (sky, black frame, fp16 overflow) gives a huge or garbage volume and the GPU dies inside the
    refiner. Clean the depth first and keep the volume a sane size; remember what was seen."""
    from comfy.ldm.moge import model as M
    cls = getattr(M, "MoGeModelV3", None)
    if cls is None or getattr(cls, "_laa_patched", False):
        return
    orig = cls._refine_logz

    def _refine_logz(self, coord, conditioning):
        logz = coord[..., 2].detach().float()
        fin = torch.isfinite(logz)
        n_bad = int((~fin).sum())
        if fin.any():
            lo, hi = float(logz[fin].amin()), float(logz[fin].amax())
        else:
            lo = hi = 0.0
        STATS.update(shape=tuple(coord.shape[1:3]), nonfinite=n_bad, logz=(lo, hi),
                     bins=int(round((hi - lo) * self.refiner_depth_resolution)) + 1)
        span = (MAX_Z_BINS - 1) / float(self.refiner_depth_resolution)
        if n_bad or hi - lo > span:
            med = float(logz[fin].median()) if fin.any() else 0.0
            logz = torch.where(fin, logz, torch.full_like(logz, med))
            a = max(lo, min(med - span / 2, hi - span))
            logz = logz.clamp(a, a + span)
            coord = torch.cat([coord[..., :2], logz.to(coord.dtype).unsqueeze(-1)], dim=-1)
            STATS["fixed"] = True
        return orig(self, coord, conditioning)

    cls._refine_logz = _refine_logz
    cls._laa_patched = True


def stats_text():
    if not STATS:
        return "refiner not reached"
    return "point map %dx%d, %d non-finite, log-depth %.2f..%.2f = %d bins%s" % (
        STATS["shape"][1], STATS["shape"][0], STATS["nonfinite"], STATS["logz"][0], STATS["logz"][1],
        STATS["bins"], " (cleaned)" if STATS.get("fixed") else "")


class MoGe(object):
    def __init__(self, model_name, log=print):
        comfy_lib.setup(log)
        _patch_refiner(log)
        import comfy.utils
        from comfy.ldm.moge.model import MoGeModel
        path = config.require_model(model_name, "geometry_estimation")
        log("loading MoGe %s" % model_name)
        self.m = MoGeModel(comfy.utils.load_torch_file(path, safe_load=True))

    def infer(self, rgb_u8, resolution_level=9, fov_x=86.0, refine_steps=3):
        """-> dict of numpy arrays at the frame resolution:
        normal (H,W,3) OpenCV axes, depth (H,W), mask (H,W) bool, intrinsics (3,3) normalised."""
        img = torch.from_numpy(rgb_u8).float().div(255.0).permute(2, 0, 1)[None]
        with torch.inference_mode():          # like ComfyUI does: no autograd graph (it kept every activation)
            r = self._infer(img, resolution_level, fov_x, refine_steps)
        o = {}
        for k in ("normal", "depth", "mask", "intrinsics", "points"):
            if k in r and r[k] is not None:
                o[k] = r[k][0].detach().float().cpu().numpy() if k != "mask" else r[k][0].cpu().numpy()
        del r
        return o

    def _infer(self, img, resolution_level, fov_x, refine_steps):
        return self.m.infer(img, resolution_level=resolution_level,
                         fov_x=None if fov_x <= 0 else float(fov_x),
                         force_projection=True, apply_mask=False, refine_steps=refine_steps)

    def release(self):
        try:
            comfy_lib.unload_all()
        except Exception:                 # GPU already in a bad state (out of memory): don't hide the real error
            pass
        try:
            from .base import free_vram
            free_vram()
        except Exception:
            pass


def fallbacks(resolution_level, refine_steps):
    """Lighter settings to try, in order, when the GPU runs out of memory."""
    out, r, lv = [], refine_steps, resolution_level
    while r > 1:
        r -= 1
        out.append((lv, r))
    while lv > 3:
        lv = max(3, lv - 3)
        out.append((lv, r))
    out.append((lv, 0))                   # no refiner at all (MoGe-2 like)
    return out


CV_TO_GL = np.array([1.0, -1.0, -1.0], np.float32)     # OpenCV camera -> Nuke/OpenGL camera
