"""
Cosmos DiffusionRenderer (NVIDIA, inverse renderer): basecolor (albedo), roughness, metallic,
normal, depth for 57-frame video chunks, temporally consistent.

It only runs on Linux, so it lives in WSL (cosmos/setup_cosmos.bat installs it). This module:
  1. letterboxes the plate to the model's 1280x704 (no stretch) into <cache>/cosmos_in,
  2. runs cosmos/laa_cosmos.py inside WSL on those frames,
  3. un-letterboxes the outputs back to the plate size.
"""

import os
import re
import subprocess
import sys
import json

import cv2
import numpy as np

from .. import config

W_IN, H_IN = 1280, 704
DISTRO = "Ubuntu-22.04"
PASSES_ALL = ("basecolor", "roughness", "metallic", "normal", "depth")


def to_wsl(path):
    """C:\\Users\\x -> /mnt/c/Users/x"""
    p = os.path.abspath(path).replace("\\", "/")
    m = re.match(r"^([A-Za-z]):/(.*)$", p)
    return "/mnt/%s/%s" % (m.group(1).lower(), m.group(2)) if m else p


def letterbox(rgb_u8):
    """-> (1280x704 image, box (x, y, w, h) of the plate inside it)."""
    H, W = rgb_u8.shape[:2]
    s = min(W_IN / float(W), H_IN / float(H))
    w, h = int(round(W * s)), int(round(H * s))
    img = cv2.resize(rgb_u8, (w, h), interpolation=cv2.INTER_AREA)
    x, y = (W_IN - w) // 2, (H_IN - h) // 2
    img = cv2.copyMakeBorder(img, y, H_IN - h - y, x, W_IN - w - x, cv2.BORDER_REFLECT_101)
    return img, (x, y, w, h)


def unletterbox(img, box, H, W):
    x, y, w, h = box
    return cv2.resize(img[y:y + h, x:x + w], (W, H), interpolation=cv2.INTER_CUBIC)


def _srgb_to_linear(x):
    return np.where(x <= 0.04045, x / 12.92, ((x + 0.055) / 1.055) ** 2.4).astype(np.float32)


def decode(name, u8):
    """model output (uint8 RGB) -> float layer(s). basecolor -> linear albedo, normal -> -1..1."""
    f = u8.astype(np.float32) / 255.0
    if name == "basecolor":
        return {"albedo": _srgb_to_linear(f)}
    if name == "normal":
        n = f * 2.0 - 1.0
        return {"Ncosmos": n / np.maximum(np.linalg.norm(n, axis=-1, keepdims=True), 1e-6)}
    if name == "depth":
        return {"depth_cosmos": f.mean(-1)}          # relative depth, for comparison only
    return {name: f.mean(-1)}                         # roughness, metallic: one channel


class CosmosRenderer(object):
    def __init__(self, s, log=print, say=None):
        self.s, self.log = s, log
        self.say = say or (lambda t: None)
        self.proc = None

    def passes(self):
        p = ["basecolor"]
        if self.s.cosmos_material:
            p += ["roughness", "metallic"]
        if self.s.cosmos_geometry:
            p += ["normal", "depth"]
        return p

    def run(self, in_dir, out_dir, progress=None, cancel=None):
        script = os.path.join(str(config.APP_DIR), "cosmos", "laa_cosmos.py")
        cmd = ('source ~/miniforge3/etc/profile.d/conda.sh && conda activate cosmos-predict1 && '
               'cd ~/cosmos/cosmos1-diffusion-renderer && '
               'CUDA_HOME=$CONDA_PREFIX PYTHONPATH=$(pwd) python -u "%s" --in_dir "%s" --out_dir "%s" '
               '--passes %s --steps %d --overlap %d --offload %d'
               % (to_wsl(script), to_wsl(in_dir), to_wsl(out_dir), " ".join(self.passes()),
                  self.s.cosmos_steps, self.s.cosmos_overlap, 1 if self.s.cosmos_offload else 0))
        self.log("[albedo] WSL: %s" % cmd)
        flags = subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0
        self.proc = subprocess.Popen(["wsl", "-d", self.s.wsl_distro, "--exec", "bash", "-lc", cmd],
                                     stdout=subprocess.PIPE, stderr=subprocess.STDOUT, creationflags=flags)
        last = []
        try:
            for raw in iter(self.proc.stdout.readline, b""):
                line = raw.decode("utf-8", "replace").rstrip()
                if cancel and cancel():
                    self.stop()
                    raise InterruptedError()
                if line.startswith("@@COSMOS "):
                    try:
                        m = json.loads(line[9:])
                    except ValueError:
                        continue
                    if m.get("t") == "status":
                        self.say(m["m"])
                    elif m.get("t") == "progress" and progress:
                        progress(m["done"], m["total"])
                    elif m.get("t") == "log":
                        self.log("[albedo] " + m["m"])
                elif line.strip():
                    last = (last + [line])[-40:]
                    if "Error" in line or "error" in line or "Traceback" in line:
                        self.log(line)
            code = self.proc.wait()
        finally:
            self.proc = None
        if code != 0:
            hint = ""
            txt = "\n".join(last)
            if "conda" in txt and ("not found" in txt or "No such file" in txt):
                hint = " Cosmos is not installed: run cosmos/setup_cosmos.bat."
            elif "There is no distribution" in txt or "WSL" in txt and "not" in txt:
                hint = " WSL / Ubuntu not found: run cosmos/install_wsl.bat then setup_cosmos.bat."
            elif "out of memory" in txt.lower():
                hint = " Out of GPU memory: keep 'Cosmos offload' on, close other GPU apps."
            raise RuntimeError("Cosmos failed (exit %s).%s\n%s" % (code, hint, txt[-2000:]))

    def stop(self):
        """kill the WSL side too: killing wsl.exe alone can leave the Linux process on the GPU."""
        try:
            subprocess.run(["wsl", "-d", self.s.wsl_distro, "--exec", "pkill", "-f", "laa_cosmos.py"],
                           timeout=10, creationflags=subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0)
        except Exception:
            pass
        if self.proc is not None:
            try:
                self.proc.kill()
            except Exception:
                pass


def kill_wsl_job(distro=DISTRO):
    """used by the UI when Cancel kills the worker process."""
    try:
        flags = subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0
        subprocess.Popen(["wsl", "-d", distro, "--exec", "pkill", "-f", "laa_cosmos.py"], creationflags=flags)
    except Exception:
        pass
