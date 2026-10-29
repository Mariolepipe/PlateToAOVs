"""
Person / head crops for Sapiens2 (same logic as the NukeBridge Sapiens2NormalCrops node).

Sapiens2 takes a 1024x768 portrait input: a 2048x870 plate is squeezed into 768x326 and a face
is ~45 px for the model. Each person (and their head) is cropped and run at native scale, then
pasted back with a feathered edge, heads over bodies. Body boxes are fixed over a chunk of frames
(no jitter), head boxes follow the head, smoothed in time.
"""

import numpy as np
import torch
import torch.nn.functional as F

try:
    from scipy import ndimage as _ndi
except Exception:
    _ndi = None

ASPECT = 768.0 / 1024.0      # Sapiens2 input width / height


def bbox(m):
    ys, xs = np.where(m)
    if len(xs) == 0:
        return None
    return [int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1]


def fit(box, W, H, pad):
    """pad the box, grow it to the Sapiens aspect, keep it inside the frame when possible."""
    x0, y0, x1, y1 = box
    w, h = x1 - x0, y1 - y0
    cx, cy = (x0 + x1) / 2.0, (y0 + y1) / 2.0
    w, h = w * (1 + 2 * pad), h * (1 + 2 * pad)
    if w / h > ASPECT:
        h = w / ASPECT
    else:
        w = h * ASPECT
    w, h = min(w, W), min(h, H)
    x0 = int(round(min(max(cx - w / 2, 0), W - w)))
    y0 = int(round(min(max(cy - h / 2, 0), H - h)))
    return [x0, y0, x0 + int(round(w)), y0 + int(round(h))]


def feather(box, W, H, f):
    """alpha ramp on the crop sides that are not on the frame border (smoothstep)."""
    x0, y0, x1, y1 = box
    h, w = y1 - y0, x1 - x0
    f = max(int(f), 1)
    ax = np.ones(w, np.float32)
    ay = np.ones(h, np.float32)
    r = np.clip((np.arange(w, dtype=np.float32) + 0.5) / f, 0, 1)
    if x0 > 0:
        ax = np.minimum(ax, r)
    if x1 < W:
        ax = np.minimum(ax, r[::-1])
    r = np.clip((np.arange(h, dtype=np.float32) + 0.5) / f, 0, 1)
    if y0 > 0:
        ay = np.minimum(ay, r)
    if y1 < H:
        ay = np.minimum(ay, r[::-1])
    a = ay[:, None] * ax[None, :]
    return a * a * (3 - 2 * a)


def components(union, W, H, min_area=0.002):
    if _ndi is not None:
        lab, n = _ndi.label(union)
        comps = [lab == k for k in range(1, n + 1)]
    else:
        comps = [union]
    return [c for c in comps if c.sum() >= min_area * W * H]


def compute_crops(union, W, H, pad=0.08, head=True, head_frac=0.3, min_area=0.002, head_union=None, comps=None):
    """-> (boxes, n_body, comps): body crops first then head crops (fallback head boxes).
    comps: one mask per person (SAM3 numbers); default = touching blobs of union."""
    if comps is None:
        comps = components(union, W, H, min_area)
    else:
        comps = [c for c in comps if c.sum() >= min_area * W * H * 0.25]
    bodies, heads = [], []
    for c in comps:
        b = bbox(c)
        bodies.append(fit(b, W, H, pad))
        if head:
            hb = None
            if head_union is not None:
                hp = c & head_union
                if hp.sum() >= 0.0002 * W * H:
                    hb = bbox(hp)
            if hb is None:
                top = b[1] + max(int((b[3] - b[1]) * head_frac), 8)
                hm = c.copy()
                hm[top:] = False
                hb = bbox(hm)
            if hb is not None:
                heads.append(fit(hb, W, H, pad * 2))
    return bodies + heads, len(bodies), comps


def track_heads(comps, head_frames, W, H, pad, smooth=1.5):
    """Per-frame head crops that follow the head: fixed size, centre tracked and smoothed in time."""
    B = head_frames.shape[0]
    tracks = []
    for c in comps:
        boxes = [bbox(head_frames[i] & c) for i in range(B)]
        valid = [b for b in boxes if b is not None and (b[2] - b[0]) * (b[3] - b[1]) >= 0.0001 * W * H]
        if not valid:
            tracks.append(None)
            continue
        sw = max(b[2] - b[0] for b in valid)
        sh = max(b[3] - b[1] for b in valid)
        cx = np.array([np.nan if b is None else (b[0] + b[2]) / 2.0 for b in boxes])
        cy = np.array([np.nan if b is None else (b[1] + b[3]) / 2.0 for b in boxes])
        idx = np.arange(B)
        ok = ~np.isnan(cx)
        cx = np.interp(idx, idx[ok], cx[ok])
        cy = np.interp(idx, idx[ok], cy[ok])
        if smooth > 0 and B > 1:
            r = int(3 * smooth)
            k = np.exp(-0.5 * (np.arange(-r, r + 1) / smooth) ** 2)
            k /= k.sum()
            cx = np.convolve(np.pad(cx, r, mode="edge"), k, mode="valid")
            cy = np.convolve(np.pad(cy, r, mode="edge"), k, mode="valid")
        tracks.append([fit([x - sw / 2, y - sh / 2, x + sw / 2, y + sh / 2], W, H, pad) for x, y in zip(cx, cy)])
    return tracks


def chunk_boxes(person, head, pad=0.08, head_pass=True, head_frac=0.3, ids=None):
    """person, head: (B, H, W) bool masks of one chunk -> (per-frame box lists, n_body).
    ids: (B, H, W) person numbers (SAM3): one body + head crop per number, even when people touch."""
    B, H, W = person.shape
    comps = None
    if ids is not None:
        # the whole numbered person, with what you included in them (a gun held out...): the crop
        # must not cut it; Sapiens normals are still only used on the body (merge mask)
        comps = [(ids == k).any(0) for k in np.unique(ids) if k]
    boxes, n_body, comps = compute_crops(person.any(0), W, H, pad, head_pass, head_frac,
                                         head_union=None if head is None else head.any(0), comps=comps)
    frame_boxes = [list(boxes) for _ in range(B)]
    if head_pass and head is not None:
        tracks = track_heads(comps, head, W, H, pad * 2)
        fallback = boxes[n_body:]
        for i in range(B):
            if len(fallback) == len(tracks):
                heads_i = [t[i] if t is not None else fb for t, fb in zip(tracks, fallback)]
            else:
                heads_i = [t[i] for t in tracks if t is not None]
            frame_boxes[i] = boxes[:n_body] + heads_i
    return frame_boxes, n_body


def paste(results, boxes, W, H, feather_px, channels=3, neutral=None):
    """results: per-box arrays (C, h, w) torch -> (C, H, W) blended with the crop feathers."""
    acc = torch.zeros(channels, H, W)
    if neutral is not None:
        for c, v in enumerate(neutral):
            acc[c] = v
    for n, b in zip(results, boxes):
        x0, y0, x1, y1 = b
        a = torch.from_numpy(feather(b, W, H, feather_px))
        acc[:, y0:y1, x0:x1] = acc[:, y0:y1, x0:x1] * (1 - a) + n * a
    return acc


def soft_mask(m, erode, blur):
    """(B,H,W) bool/float -> eroded + blurred float mask (Sapiens inside, MoGe outside)."""
    t = torch.as_tensor(m).float()[:, None]
    if erode > 0:
        t = -F.max_pool2d(-t, 2 * erode + 1, stride=1, padding=erode)
    if blur > 0:
        k = 2 * blur + 1
        t = F.avg_pool2d(F.avg_pool2d(t, k, 1, blur, count_include_pad=False), k, 1, blur, count_include_pad=False)
    return t[:, 0].clamp(0, 1).numpy()


# ------------------------------------------------------------------ video head tracks (Cosmos)
def head_tracks(head_masks, W, H, aspect=16.0 / 9.0, head_share=0.5, min_area=0.0004,
                max_jump=0.12, min_len=0.25, smooth=2.0):
    """
    Follow every head through the sequence for a VIDEO model (Cosmos): one fixed-size box per
    head, in the model's aspect, centred on the head and smoothed in time.

    head_masks : (N, H, W) bool, head pixels (face, hair...) per frame
    head_share : head height / box height (0.5 = the head fills half the crop)
    -> list of tracks, each {"boxes": [ [x0,y0,x1,y1] per frame ], "present": [bool per frame]}
    """
    N = head_masks.shape[0]
    dets = []                                       # per frame: [(cx, cy, h, w)]
    for i in range(N):
        if _ndi is not None:
            lab, n = _ndi.label(head_masks[i])
            blobs = [lab == k for k in range(1, n + 1)]
        else:
            blobs = [head_masks[i]]
        d = []
        for b in blobs:
            if b.sum() < min_area * W * H:
                continue
            x0, y0, x1, y1 = bbox(b)
            d.append(((x0 + x1) / 2.0, (y0 + y1) / 2.0, float(y1 - y0), float(x1 - x0)))
        dets.append(d)

    tracks = []                                     # {"c": [(cx,cy) or None]*N, "s": [h]}
    for i in range(N):
        used = set()
        for t in tracks:                            # greedy: closest detection to the last position
            last = next((c for c in reversed(t["c"][:i]) if c is not None), None)
            best, bd = None, max_jump * W
            for k, (cx, cy, h, w) in enumerate(dets[i]):
                if k in used or last is None:
                    continue
                dd = ((cx - last[0]) ** 2 + (cy - last[1]) ** 2) ** 0.5
                if dd < bd:
                    best, bd = k, dd
            if best is not None:
                used.add(best)
                cx, cy, h, w = dets[i][best]
                t["c"][i] = (cx, cy)
                t["s"].append(max(h, w / 0.8))
        for k, (cx, cy, h, w) in enumerate(dets[i]):
            if k not in used:
                t = {"c": [None] * N, "s": [max(h, w / 0.8)]}
                t["c"][i] = (cx, cy)
                tracks.append(t)

    out = []
    for t in tracks:
        present = [c is not None for c in t["c"]]
        if sum(present) < max(2, min_len * N) and not (N <= 3 and any(present)):
            continue
        idx = np.arange(N)
        ok = np.array(present)
        cx = np.interp(idx, idx[ok], [c[0] for c in t["c"] if c is not None])
        cy = np.interp(idx, idx[ok], [c[1] for c in t["c"] if c is not None])
        if smooth > 0 and N > 1:
            r = int(3 * smooth)
            k = np.exp(-0.5 * (np.arange(-r, r + 1) / smooth) ** 2)
            k /= k.sum()
            cx = np.convolve(np.pad(cx, r, mode="edge"), k, mode="valid")
            cy = np.convolve(np.pad(cy, r, mode="edge"), k, mode="valid")
        bh = min(float(H), float(np.percentile(t["s"], 90)) / head_share)
        bw = min(float(W), bh * aspect)
        bh = bw / aspect
        boxes = []
        for x, y in zip(cx, cy):
            x0 = int(round(min(max(x - bw / 2, 0), W - bw)))
            y0 = int(round(min(max(y - bh * 0.42, 0), H - bh)))      # head a bit above the centre
            boxes.append([x0, y0, x0 + int(round(bw)), y0 + int(round(bh))])
        out.append({"boxes": boxes, "present": present})
    return out

