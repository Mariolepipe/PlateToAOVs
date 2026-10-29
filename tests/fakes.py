"""Fake models (no weights, CPU) for the tests. SLOW=seconds makes each fake frame slow (cancel test)."""
import os, sys, tempfile, shutil, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, torch, time
SLOW = float(os.environ.get('LAA_FAKE_SLOW', '0'))

from laa import config
from laa.io import sequence as seq
import laa.models.sapiens2 as S
import laa.models.moge as M
import laa.models.vda as V
import laa.models.ltx_alpha as L

H, W, N = 180, 420, 7
def person_mask(i):
    yy, xx = np.mgrid[0:H, 0:W]
    cx = 150 + 6 * i
    body = ((xx - cx) / 35.0) ** 2 + ((yy - 115) / 60.0) ** 2 < 1
    head = ((xx - cx) / 16.0) ** 2 + ((yy - 40) / 18.0) ** 2 < 1
    return body, head

class FakeSap:
    def __init__(self, task, size="1b", log=print): self.task = task
    def to_gpu(self): pass
    def to_cpu(self): pass
    def segment(self, rgb):
        time.sleep(SLOW)
        # recover the frame index from the red channel of pixel (0,0)
        i = int(rgb[0, 0, 0]) // 20
        b, h = person_mask(i)
        c = np.zeros((H, W), np.uint8); c[b] = 22; c[h] = 3
        return c
    def normals(self, crop):
        h, w = crop.shape[:2]
        n = torch.zeros(3, h, w); n[0] = -0.3; n[2] = 1  # Sapiens: X flipped vs MoGe
        return n / n.norm(dim=0, keepdim=True)
    def matte(self, crop):
        return (crop[..., 1] > 100).astype(np.float32)

class FakeMoGe:
    def __init__(self, name, log=print): pass
    def infer(self, rgb, *a):
        n = np.zeros((H, W, 3), np.float32); n[..., 0] = 0.3; n[..., 2] = -1   # OpenCV: -Z = towards camera
        n /= np.linalg.norm(n, axis=-1, keepdims=True)
        K = np.array([[0.53, 0, .5], [0, 1.25, .5], [0, 0, 1]], np.float32)
        return {"normal": n, "depth": np.full((H, W), 4.0, np.float32), "mask": np.ones((H, W), bool), "intrinsics": K}
    def release(self): pass

class FakeVDA:
    def __init__(self, name, log=print): pass
    def infer(self, frames, *a, progress=None, **k):
        return np.full((len(frames), H // 2, W // 2), 5.0, np.float32)

class FakeLTX:
    def __init__(self, s, log=print, say=None): (say or print)("Loading LTX (fake)")
    def matte(self, frames, progress=None, label=""):
        progress and progress(0.5)
        assert len(frames) <= 25 + 4
        return (frames[..., 1] > 100).astype(np.float32)
    def release(self): pass

class FakeIID:
    def __init__(self, variant="appearance", steps=4, log=print): pass
    def to_gpu(self): pass
    def to_cpu(self): pass
    def infer(self, rgb, seed=0):
        return {"albedo": np.full(rgb.shape, 0.5, np.float32), "roughness": np.full(rgb.shape[:2], 0.3, np.float32)}

import laa.models.albedo as AL
import laa.models.cosmos as CO


class FakeCosmos(CO.CosmosRenderer):
    """writes what laa_cosmos.py would: <out>/<pass>/<i>.png, here = the letterboxed input itself"""
    def run(self, in_dir, out_dir, progress=None, cancel=None):
        from PIL import Image
        files = sorted(os.listdir(in_dir))
        for k, p in enumerate(self.passes()):
            os.makedirs(os.path.join(out_dir, p), exist_ok=True)
            for f in files:
                im = np.asarray(Image.open(os.path.join(in_dir, f)))
                assert im.shape == (704, 1280, 3), im.shape
                Image.fromarray(im).save(os.path.join(out_dir, p, f))
            progress and progress(k + 1, len(self.passes()))
import laa.models.matanyone2 as MA


class FakeMatAnyone:
    calls = []

    def __init__(self, log=print, ckpt=None): log("loading MatAnyone 2 (fake)")

    def matte(self, frames_u8, masks, anchor, warmup=10, grow=10, check=None, progress=None):
        FakeMatAnyone.calls.append(("matte", int(masks[anchor].sum() > 0), anchor))
        progress and progress(len(masks), len(masks))
        return masks.astype(np.float32)

    def release(self): pass


def install_fakes():
    MA.MatAnyone2 = FakeMatAnyone
    S.Sapiens2 = FakeSap; M.MoGe = FakeMoGe; V.VDA = FakeVDA; L.LTXAlpha = FakeLTX; AL.MarigoldIID = FakeIID; CO.CosmosRenderer = FakeCosmos
    import laa.models.cosmos_native as CN
    CN.CosmosNative = FakeCosmos
install_fakes()



def make_plate(folder):
    os.makedirs(folder, exist_ok=True)
    for i in range(N):
        img = np.full((H, W, 3), 40, np.uint8); img[0, 0, 0] = i * 20
        b, h = person_mask(i); img[b | h] = (200, 180, 160)
        seq.write_png(os.path.join(folder, "shot_010.%04d.png" % (1001 + i)), img)
    return os.path.join(folder, "shot_010.####.png")
