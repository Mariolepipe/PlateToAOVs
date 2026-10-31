"""Visual checks written next to the passes (what the ComfyUI debug workflow showed)."""

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from .crops import ASPECT


def normals_u8(n):
    """(H,W,3) -1..1 -> uint8 0..255 (usual normal-map encoding)."""
    return np.clip((n + 1.0) * 127.5 + 0.5, 0, 255).astype(np.uint8)


def gray_u8(a, lo=None, hi=None):
    a = np.asarray(a, np.float32)
    if lo is None:
        v = a[np.isfinite(a)]
        lo, hi = (np.percentile(v, 1), np.percentile(v, 99)) if v.size else (0.0, 1.0)
    g = np.clip((a - lo) / max(hi - lo, 1e-6), 0, 1)
    return (np.repeat(g[..., None], 3, -1) * 255 + 0.5).astype(np.uint8)


def depth_u8(d):
    """inverse depth, percentile normalised (near = bright)."""
    d = np.asarray(d, np.float32)
    ok = np.isfinite(d) & (d > 0)
    inv = np.where(ok, 1.0 / np.maximum(d, 1e-6), 0)
    if not ok.any():
        return np.zeros(d.shape + (3,), np.uint8)
    lo, hi = np.percentile(inv[ok], 1), np.percentile(inv[ok], 99)
    return gray_u8(inv, lo, hi)


def draw_boxes(rgb_u8, boxes, n_body, mask=None):
    """plate + person mask tint (blue) + crop boxes (green = body, amber = head) with their number."""
    a = rgb_u8.copy()
    if mask is not None:
        a[mask] = (a[mask] * 0.6 + np.array([60, 90, 255]) * 0.4).astype(np.uint8)
    lw = max(2, a.shape[1] // 400)
    for k, (x0, y0, x1, y1) in enumerate(boxes):
        col = (61, 220, 151) if k < n_body else (255, 186, 73)
        cv2.rectangle(a, (x0, y0), (x1 - 1, y1 - 1), col, lw)
        cv2.rectangle(a, (x0, y0), (x0 + 34, y0 + 26), col, -1)
        cv2.putText(a, str(k + 1), (x0 + 6, y0 + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
    return a


def tile_sheet(crops, th=384, aspect=None):
    """one tile per crop: plate crop on top, model output of that crop below.
    aspect=None: Sapiens 3:4 tiles; 'keep': each tile keeps its crop's own shape."""
    cols = []
    for rgb_u8, n in crops:
        if aspect == "keep":
            tw = max(8, int(round(th * rgb_u8.shape[1] / float(rgb_u8.shape[0]))))
        else:
            tw = int(th * (aspect or ASPECT))
        r = torch.from_numpy(rgb_u8).float().div(255).permute(2, 0, 1)[None]
        r = F.interpolate(r, size=(th, tw), mode="bilinear", align_corners=False)[0]
        if n.shape[0] == 1:
            n = n.expand(3, -1, -1)
            nn_ = F.interpolate(n[None].float(), size=(th, tw), mode="bilinear", align_corners=False)[0]
        else:
            nn_ = F.interpolate(((n + 1) * 0.5)[None].float(), size=(th, tw), mode="bilinear", align_corners=False)[0]
        col = F.pad(torch.cat((r, nn_), 1), (4, 4, 4, 4), value=0.05)
        cols.append(col)
    if not cols:
        return np.zeros((2 * th, int(th * ASPECT), 3), np.uint8)
    return (torch.cat(cols, 2).permute(1, 2, 0).clamp(0, 1).numpy() * 255 + 0.5).astype(np.uint8)


def angle_heatmap(a, b, mask=None, max_deg=20.0):
    """angle between two normal maps -> (heatmap uint8, mean, p95). black = identical."""
    a = a / np.maximum(np.linalg.norm(a, axis=-1, keepdims=True), 1e-6)
    b = b / np.maximum(np.linalg.norm(b, axis=-1, keepdims=True), 1e-6)
    ang = np.degrees(np.arccos(np.clip((a * b).sum(-1), -1, 1)))
    sel = np.ones(ang.shape, bool) if mask is None else (mask > 0.5)
    vals = ang[sel]
    mean = float(vals.mean()) if vals.size else float("nan")
    p95 = float(np.percentile(vals, 95)) if vals.size > 10 else float("nan")
    h = np.clip(ang / max_deg, 0, 1)
    rgb = np.stack((np.clip(h * 3, 0, 1), np.clip(h * 3 - 1, 0, 1), np.clip(h * 3 - 2, 0, 1)), -1)
    rgb = np.where(sel[..., None], rgb, 0.12)
    img = (rgb * 255 + 0.5).astype(np.uint8)
    txt = "mean %.2f deg | 95%% < %.2f deg" % (mean, p95)
    s = max(0.6, img.shape[0] / 900.0)
    cv2.rectangle(img, (0, 0), (int(560 * s), int(44 * s)), (0, 0, 0), -1)
    cv2.putText(img, txt, (int(10 * s), int(30 * s)), cv2.FONT_HERSHEY_SIMPLEX, 0.8 * s, (255, 255, 255), max(1, int(2 * s)))
    return img, mean, p95
