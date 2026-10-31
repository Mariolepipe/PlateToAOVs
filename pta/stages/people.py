"""
Who is who: one number per person for the whole shot, as in Sammie-Roto.

  Find everyone : SAM 3.1 finds the people (text "person"); each one starts SAM 2.1 with their mask
                  on the frame where they are the most visible.
  Your clicks   : keyframes. On a frame, for a person: left click = this person, right click = not
                  this person. SAM 2.1 cuts that frame right away (live preview).
  Propagate     : SAM 2.1 follows everyone through the shot using ALL the keyframes at once.

Edits (pta_people.json next to the EXRs), ids are the numbers shown in the viewer:
  {"base": ..., "auto": true | false, "removed": [3], "same": {"4": 1},
   "keys": [{"frame": 12, "target": 2, "points": [[x, y, 1 | 0], ...],
             "base": "pta_people_keys/f00012_p2.png" | null}],
   "validated": true}
points are in plate pixels. "base" is the person's mask on that frame when you started clicking
(working size), so the clicks replay exactly the same later.
"""

import numpy as np

MIN_PX = 64                 # a mask smaller than this (working size) is "not there"

# well separated colours, number k uses PALETTE[(k - 1) % len]
PALETTE = [(255, 99, 71), (64, 156, 255), (50, 205, 120), (255, 196, 0), (186, 104, 255),
           (0, 210, 210), (255, 120, 190), (160, 200, 60), (255, 150, 40), (120, 140, 255),
           (230, 80, 80), (80, 220, 170), (200, 160, 255), (255, 220, 120), (90, 180, 255), (210, 120, 60)]


def color(k):
    return PALETTE[(int(k) - 1) % len(PALETTE)]


def empty_edits(base=None):
    return {"base": base, "removed": [], "same": {}, "keys": [], "validated": False}


# ------------------------------------------------------------------ SAM3 + fixes
def anchors(A, min_px=None):
    """SAM3 result (N, O, h, w) -> [(frame, person, mask)]: each person on the frame where they are
    the biggest (the clearest start for SAM2)."""
    out = []
    for k in range(A.shape[1]):
        areas = A[:, k].reshape(A.shape[0], -1).sum(1)
        f = int(np.argmax(areas))
        if areas[f] >= (MIN_PX if min_px is None else min_px):
            out.append((f, k + 1, A[f, k]))
    return out


def scaled_points(key, scale):
    return [(x * scale, y * scale, int(l)) for x, y, l in key.get("points", [])]


def masks_from_propagation(out, N, h, w):
    """SAM2 {frame: {person: mask}} -> (N, O, h, w) bool, object k = person k + 1"""
    n = max([o for d in out.values() for o in d] or [0])
    A = np.zeros((N, n, h, w), bool)
    for f, d in out.items():
        for o, m in d.items():
            if 0 <= f < N and m.shape == (h, w):
                A[f, o - 1] = m
    return A


def keyed_top(keys):
    """people you clicked on win where people overlap, the last clicked first"""
    top = []
    for k in reversed(keys or []):
        t = int(k.get("target", 0)) - 1
        if t >= 0 and t not in top:
            top.append(t)
    return top


def to_id_maps(masks, top=()):
    """(N, O, h, w) bool -> (N, h, w) uint8. Where two people overlap: the people you fixed win
    (top, first = most recent), otherwise the smaller one wins (the visible part of someone behind
    is small, the one in front covers it)."""
    N, O = masks.shape[:2]
    top = [k for k in top if 0 <= k < O]
    out = np.zeros((N,) + masks.shape[2:], np.uint8)
    for i in range(N):
        areas = masks[i].reshape(O, -1).sum(1)
        order = [k for k in np.argsort(-areas) if k not in top] + list(reversed(top))
        for k in order:
            if areas[k] >= MIN_PX:
                out[i][masks[i, k]] = k + 1
    return out


# ------------------------------------------------------------------ review decisions
def lut(edits, n_ids=255):
    """id -> final id (0 removed). 'same' chains are followed: 5 same as 4, 4 same as 1 -> 1."""
    t = np.arange(256, dtype=np.uint8)
    same = {int(k): int(v) for k, v in (edits or {}).get("same", {}).items()}
    for k in range(1, n_ids + 1):
        v, seen = k, set()
        while v in same and v not in seen:
            seen.add(v)
            v = same[v]
        t[k] = v
    for k in (edits or {}).get("removed", []):
        t[t == int(k)] = 0
    t[0] = 0
    return t


# ------------------------------------------------------------------ boxes
def overlay(rgb, ids, alpha=0.5):
    """plate (h, w, 3) uint8 + final ids (any size) -> coloured people with their number."""
    import cv2
    h, w = rgb.shape[:2]
    if ids.shape != (h, w):
        ids = cv2.resize(ids, (w, h), interpolation=cv2.INTER_NEAREST)
    out = (rgb.astype(np.float32) * 0.55)
    for k in [int(v) for v in np.unique(ids) if v]:
        m = ids == k
        out[m] = out[m] * (1 - alpha) + np.array(color(k), np.float32) * alpha + rgb[m] * 0.45 * (1 - alpha)
    out = np.clip(out, 0, 255).astype(np.uint8)
    for k in [int(v) for v in np.unique(ids) if v]:
        m = (ids == k).astype(np.uint8)
        cnt, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(out, cnt, -1, color(k), max(1, w // 700), cv2.LINE_AA)
        ys, xs = np.where(m)
        # label on the upper part of the person (the head area), inside the mask
        top = ys.min() + (ys.max() - ys.min()) * 0.25
        sel = ys <= top
        cx, cy = int(np.median(xs[sel])), int(np.median(ys[sel]))
        r = max(12, w // 60)
        cv2.circle(out, (cx, cy), r, (20, 20, 22), -1, cv2.LINE_AA)
        cv2.circle(out, (cx, cy), r, color(k), max(2, r // 6), cv2.LINE_AA)
        txt = str(k)
        fs = r / 22.0
        (tw, th), _ = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, fs, max(1, int(fs * 2)))
        cv2.putText(out, txt, (cx - tw // 2, cy + th // 2), cv2.FONT_HERSHEY_SIMPLEX, fs, (255, 255, 255),
                    max(1, int(fs * 2)), cv2.LINE_AA)
    return out
