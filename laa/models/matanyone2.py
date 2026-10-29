"""
MatAnyone 2 (NTU S-Lab, CVPR 2026): video matting guided by a mask. Given the SAM 2.1 mask of
one number on one frame, it follows that person / object through the shot and gives a soft alpha
(hair, motion blur) of that one only. The official code is cloned into vendor/MatAnyone2.

License of MatAnyone 2: S-Lab License 1.0, non-commercial use (see vendor/MatAnyone2/LICENSE.txt).
"""

import os
import sys

import numpy as np
import torch

from .. import config
from .base import device, free_vram

CKPT = "matanyone2.pth"
CKPT_URL = "https://github.com/pq-yang/MatAnyone2/releases/download/v1.0.0/matanyone2.pth"
REPO = "https://github.com/pq-yang/MatAnyone2.git"


def repo_dir():
    return os.path.join(str(config.APP_DIR), "vendor", "MatAnyone2")


def _code():
    d = repo_dir()
    if not os.path.isfile(os.path.join(d, "matanyone2", "inference", "inference_core.py")):
        raise RuntimeError("MatAnyone 2 code not found in %s. Download it in the app: Models, bottom right." % d)
    if d not in sys.path:
        sys.path.insert(0, d)
    try:                                      # their inference module imports imageio for a helper we do not use
        import imageio  # noqa: F401  (pyflakes: only checks that it is there)
    except ImportError:                       # not installed: a stand-in only while their code is imported,
        import types                          # then removed (diffusers, for Cosmos, checks imageio too)
        sys.modules["imageio"] = types.ModuleType("imageio")
        try:
            import matanyone2.inference.inference_core  # noqa: F401
        finally:
            sys.modules.pop("imageio", None)
    return d


def _cfg(d):
    """the official config without Hydra (SAM 2.1 already uses Hydra in this process)"""
    from omegaconf import OmegaConf, open_dict
    c = os.path.join(d, "matanyone2", "config")
    cfg = OmegaConf.load(os.path.join(c, "eval_matanyone_config.yaml"))
    for k in ("defaults", "hydra"):
        cfg.pop(k, None)
    with open_dict(cfg):
        cfg.model = OmegaConf.load(os.path.join(c, "model", "base.yaml"))
        cfg.model.pretrained_resnet = False    # the checkpoint has every weight: no ImageNet download
    return cfg


class MatAnyone2(object):
    def __init__(self, log=print, ckpt=None):
        d = _code()
        from matanyone2.model.matanyone2 import MatAnyone2 as Net
        self.dev = device()
        path = ckpt or config.require_model(CKPT, "matanyone", "Download it in the app: Models, bottom right.")
        log("loading MatAnyone 2 %s" % path)
        self.cfg = _cfg(d)
        self.cfg.weights = path
        with torch.inference_mode():
            self.net = Net(self.cfg, single_object=True).to(self.dev).eval()
            self.net.load_weights(torch.load(path, map_location=self.dev))

    def _run(self, frames, mask, warmup, check=None):
        """frames (T,3,h,w) float 0..1 on device, mask (h,w) float 0/1 on frame 0 -> (T,h,w) alpha"""
        from matanyone2.inference.inference_core import InferenceCore
        proc = InferenceCore(self.net, cfg=self.cfg, device=self.dev)
        out = []
        for ti in range(warmup + frames.shape[0]):
            if check:
                check()
            j = max(0, ti - warmup)
            img = frames[j]
            if ti == 0:
                proc.step(img, mask, objects=[1])
                p = proc.step(img, first_frame_pred=True)
            elif ti <= warmup:
                p = proc.step(img, first_frame_pred=True)   # same first frame again: settle the alpha
            else:
                p = proc.step(img)
            if ti >= warmup:
                out.append(proc.output_prob_to_mask(p).float().clamp(0, 1).cpu())
        del proc
        return torch.stack(out).numpy()

    def matte(self, frames_u8, masks, anchor, warmup=10, grow=10, check=None, progress=None):
        """
        frames_u8: (N, h, w, 3) uint8 plate, masks: (N, h, w) bool SAM mask of ONE number, both at
        the matting size. anchor: its clearest frame; forward to the end, then backward to frame 0.
        -> (N, h, w) float32 alpha of that number only
        """
        from matanyone2.utils.inference_utils import gen_dilate, gen_erosion
        from matanyone2.utils.device import safe_autocast
        N = masks.shape[0]
        m = masks[anchor].astype(np.uint8) * 255
        if grow > 0:                             # official preprocessing: closes small holes of the mask
            m = gen_erosion(gen_dilate(m, grow, grow), grow, grow)
        alpha = np.zeros(masks.shape, np.float32)
        with torch.inference_mode(), safe_autocast():
            def frames(idx):
                return torch.from_numpy(np.ascontiguousarray(frames_u8[idx])).to(self.dev).permute(0, 3, 1, 2).float() / 255.0
            mk = torch.from_numpy(np.asarray(m, np.float32)).to(self.dev)
            fw = list(range(anchor, N))           # forward from the anchor, then backward to frame 0
            bw = list(range(anchor, -1, -1))
            done = 0
            for idx in ([fw, bw] if len(bw) > 1 else [fw]):
                a = self._run(frames(idx), mk, warmup, check)
                skip = idx is bw                  # the anchor frame comes from the forward pass
                for k, i in enumerate(idx):
                    if not (skip and k == 0):
                        alpha[i] = a[k]
                done += len(idx) - (1 if skip else 0)
                if progress:
                    progress(done, N)
        return alpha

    def release(self):
        self.net = None
        free_vram()


def download(log=print):
    """vendor/MatAnyone2 (official code) + its checkpoint (140 MB)"""
    import subprocess
    import urllib.request
    d = repo_dir()
    if not os.path.isfile(os.path.join(d, "matanyone2", "inference", "inference_core.py")):
        log("cloning pq-yang/MatAnyone2 into %s" % d)
        subprocess.check_call(["git", "clone", "--depth", "1", REPO, d])
    have = config.find_model(CKPT)
    if have:
        log("already there: %s" % have)
        return have
    dst_dir = config.default_download_dir("matanyone")
    dst = os.path.join(dst_dir, CKPT)
    log("downloading %s into %s" % (CKPT, dst_dir))
    tmp = dst + ".part"

    def hook(b, bs, total):
        if total > 0 and b % 100 == 0:
            print("  %3d%%" % min(100, b * bs * 100 // total), end="\r", flush=True)
    urllib.request.urlretrieve(CKPT_URL, tmp, hook)
    os.replace(tmp, dst)
    log("done: %s" % dst)
    return dst
