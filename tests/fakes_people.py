"""Fakes for the people tests: two people, the one behind half hidden; fake SAM3 first sees them
as ONE person."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from fakes import *
import laa.models.sam3 as SM


def two(i):
    """front person A (moves right), back person B (static, partly behind A)"""
    yy, xx = np.mgrid[0:H, 0:W]
    ca, cb = 140 + 8 * i, 205
    a = (((xx - ca) / 34.0) ** 2 + ((yy - 115) / 60.0) ** 2 < 1) | (((xx - ca) / 15.0) ** 2 + ((yy - 42) / 17.0) ** 2 < 1)
    b = (((xx - cb) / 30.0) ** 2 + ((yy - 110) / 55.0) ** 2 < 1) | (((xx - cb) / 14.0) ** 2 + ((yy - 45) / 16.0) ** 2 < 1)
    head_a = ((xx - ca) / 15.0) ** 2 + ((yy - 42) / 17.0) ** 2 < 1
    head_b = (((xx - cb) / 14.0) ** 2 + ((yy - 45) / 16.0) ** 2 < 1) & ~a
    return a, b & ~a, head_a, head_b


def frame_index(img):
    return int(img[0, 0, 0]) // 20


class FakeSap2(FakeSap):
    def segment(self, rgb):
        a, b, ha, hb = two(frame_index(rgb))
        c = np.zeros((H, W), np.uint8); c[a | b] = 22; c[ha | hb] = 3
        return c


class FakeSAM3:
    calls = []

    def __init__(self, s, log=print, say=None): (say or print)("Loading SAM 3.1 (fake)")

    def track(self, frames, init_masks=None, detect=True, progress=None, cond=None):
        FakeSAM3.calls.append(("sam3", len(frames)))
        idx = [frame_index(f) for f in frames]      # SAM3 alone: the two people as ONE object
        return np.stack([[two(i)[0] | two(i)[1]] for i in idx])

    def encode(self, prompt):
        return prompt

    def release(self): pass


class FakeSAM2Video:
    """keyframes -> which true person(s) each object is -> those people on every frame"""
    calls = []

    def __init__(self, frames_dir, log=print, **kw):
        self.n = len([f for f in os.listdir(frames_dir) if f.endswith(".jpg")])
        self.who = {}                               # obj -> set of true people (0 = a, 1 = b)
        FakeSAM2Video.calls.append(("init", self.n))

    def _ident(self, m, i):
        t = two(i)[:2]
        return {k for k in (0, 1) if (m & t[k]).sum() > 0.5 * t[k].sum()}

    def _mask(self, who, i):
        t = two(i)[:2]
        m = np.zeros((H, W), bool)
        for k in who:
            m |= t[k]
        return m

    def add_mask(self, frame, obj, mask):
        self.who[obj] = self._ident(mask, frame)
        return self._mask(self.who[obj], frame)

    def set_key(self, frame, obj, base=None, points=()):
        FakeSAM2Video.calls.append(("key", frame, obj, base is not None, len(points)))
        a, b = two(frame)[:2]
        who = set(self._ident(base, frame)) if base is not None else set()
        for x, y, l in points:
            k = 1 if b[int(y), int(x)] else (0 if a[int(y), int(x)] else None)
            if k is None:
                continue
            who = who | {k} if l else who - {k}
        self.who[obj] = who
        return self._mask(who, frame)

    def objects(self):
        return sorted(self.who)

    def propagate(self, progress=None, check=lambda: None):
        FakeSAM2Video.calls.append(("propagate",))
        return {i: {o: self._mask(w, i) for o, w in self.who.items()} for i in range(self.n)}

    def release(self): pass


S.Sapiens2 = FakeSap2
SM.SAM3People = FakeSAM3
import laa.models.sam2 as S2
S2.SAM2Video = FakeSAM2Video



def make_plate2(folder):
    os.makedirs(folder, exist_ok=True)
    for i in range(N):
        img = np.full((H, W, 3), 40, np.uint8); img[0, 0, 0] = i * 20
        a, b, _, _ = two(i); img[b] = (150, 170, 190); img[a] = (200, 180, 160)
        seq.write_png(os.path.join(folder, "shot_020.%04d.png" % (1001 + i)), img)
    return os.path.join(folder, "shot_020.####.png")
