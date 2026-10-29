"""
Motion-compensated temporal stabilization (same maths as the NukeBridge node), streamed over the
whole sequence: frame t is averaged with its neighbours t-r..t+r warped onto it by optical flow
(RAFT, torchvision; OpenCV DIS as fallback). Weights = plate match after warping (occlusions ->
~0) x gaussian in time x, for normals, an anti-ghost angle test. No chunk boundaries any more.
"""

import math
from collections import OrderedDict

import numpy as np
import torch
import torch.nn.functional as F

_RAFT = {}


def _raft(dev):
    if "m" not in _RAFT:
        from torchvision.models.optical_flow import raft_large, Raft_Large_Weights
        _RAFT["m"] = raft_large(weights=Raft_Large_Weights.DEFAULT, progress=True).eval()
    return _RAFT["m"].to(dev)


def release():
    if "m" in _RAFT:
        _RAFT["m"].to("cpu")
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _flow_raft(a, b, dev):
    with torch.inference_mode():
        return _raft(dev)(a * 2 - 1, b * 2 - 1, num_flow_updates=12)[-1]


def _flow_dis(a, b):
    import cv2
    g = lambda t: (t[0].mean(0).cpu().numpy() * 255).astype(np.uint8)
    f = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_MEDIUM).calc(g(a), g(b), None)
    return torch.from_numpy(f).permute(2, 0, 1)[None].to(a.device)


def _warp(img, flow):
    _, _, H, W = img.shape
    yy, xx = torch.meshgrid(torch.arange(H, device=img.device, dtype=torch.float32),
                            torch.arange(W, device=img.device, dtype=torch.float32), indexing="ij")
    gx = (xx + flow[0, 0]) / (W - 1) * 2 - 1
    gy = (yy + flow[0, 1]) / (H - 1) * 2 - 1
    grid = torch.stack((gx, gy), -1)[None]
    out = F.grid_sample(img, grid, mode="bilinear", padding_mode="border", align_corners=True)
    inside = ((gx.abs() <= 1) & (gy.abs() <= 1)).float()[None, None]
    return out, inside


class _LRU(OrderedDict):
    def __init__(self, n, fn):
        super().__init__()
        self.n, self.fn = n, fn

    def get_(self, k):
        if k in self:
            self.move_to_end(k)
            return self[k]
        v = self.fn(k)
        self[k] = v
        if len(self) > self.n:
            self.popitem(last=False)
        return v


def stabilize_stream(n_frames, get_value, get_plate, put, radius=2, sigma=0.04, max_angle=15.0,
                     normals=True, flow_width=1024, progress=None, cancel=None, log=print):
    """
    get_value(i) -> (H,W,C) float32 numpy (normals in -1..1)
    get_plate(i) -> (H,W,3) float32 0..1
    put(i, result (H,W,C) float32)
    """
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_raft = True
    try:
        _raft(dev)
    except Exception as e:
        log("RAFT unavailable (%s), using OpenCV DIS flow" % e)
        use_raft = False

    def load(i):
        v = torch.from_numpy(np.ascontiguousarray(get_value(i), np.float32)).permute(2, 0, 1)[None].to(dev)
        p = torch.from_numpy(np.ascontiguousarray(get_plate(i), np.float32)).permute(2, 0, 1)[None]
        H, W = p.shape[-2:]
        s = min(1.0, flow_width / float(W))
        fw, fh = max(8, int(round(W * s / 8)) * 8), max(8, int(round(H * s / 8)) * 8)
        small = F.interpolate(p, size=(fh, fw), mode="bilinear", align_corners=False).to(dev)
        lum = F.avg_pool2d(p.mean(1, keepdim=True), 5, 1, 2, count_include_pad=False).to(dev)
        return v, small, lum, (W / float(fw), H / float(fh))

    cache = _LRU(2 * radius + 2, load)
    try:
        for t in range(n_frames):
            if cancel is not None and cancel():
                raise InterruptedError()
            vt, st, lt, (sx, sy) = cache.get_(t)
            if radius <= 0 or n_frames < 2:
                put(t, vt[0].permute(1, 2, 0).cpu().numpy())
                continue
            H, W = vt.shape[-2:]
            acc = vt.clone()
            wsum = torch.ones(1, 1, H, W, device=dev)
            for k in range(-radius, radius + 1):
                j = t + k
                if k == 0 or j < 0 or j >= n_frames:
                    continue
                vj, sj, lj, _ = cache.get_(j)
                fl = _flow_raft(st, sj, dev) if use_raft else _flow_dis(st, sj)
                fl = F.interpolate(fl, size=(H, W), mode="bilinear", align_corners=False)
                fl = torch.cat((fl[:, :1] * sx, fl[:, 1:] * sy), 1)
                wv, inside = _warp(vj, fl)
                wl, _ = _warp(lj, fl)
                err = (wl - lt).abs()
                w = torch.exp(-(err / sigma) ** 2) * inside * math.exp(-0.5 * (k / max(radius, 1)) ** 2)
                if normals and max_angle > 0:
                    cosang = (wv * vt).sum(1, keepdim=True)
                    ang2 = (2.0 * (1.0 - cosang)).clamp(min=0) * (180.0 / math.pi) ** 2
                    w = w * torch.exp(-ang2 / (max_angle * max_angle))
                acc += wv * w
                wsum += w
            r = acc / wsum
            if normals:
                r = r / torch.norm(r, dim=1, keepdim=True).clamp(min=1e-6)
            put(t, r[0].permute(1, 2, 0).cpu().numpy())
            if progress:
                progress(t + 1, n_frames)
    finally:
        release()
