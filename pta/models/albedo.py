"""
Albedo (experimental): Marigold intrinsic image decomposition (ETH Zurich), via diffusers.

Sapiens2 has an albedo config but Meta did not release those weights, so the albedo comes from
Marigold IID on the whole frame. Two checkpoints:
  appearance : albedo + roughness + metallicity
  lighting   : albedo + diffuse shading + non-diffuse residual
Stored linear in the EXR (Marigold predicts sRGB-encoded albedo for the appearance model).
"""

import numpy as np
import torch

from .base import device, free_vram

REPOS = {
    "appearance": "prs-eth/marigold-iid-appearance-v1-1",
    "lighting": "prs-eth/marigold-iid-lighting-v1-1",
}


def srgb_to_linear(x):
    x = np.clip(x, 0, 1)
    return np.where(x <= 0.04045, x / 12.92, ((x + 0.055) / 1.055) ** 2.4).astype(np.float32)


class MarigoldIID(object):
    def __init__(self, variant="appearance", steps=4, log=print):
        from diffusers import MarigoldIntrinsicsPipeline
        dev = device()
        dtype = torch.float16 if dev.type == "cuda" else torch.float32
        log("loading Marigold IID %s (downloaded on first use)" % variant)
        kw = {"variant": "fp16"} if dtype == torch.float16 else {}
        try:
            self.pipe = MarigoldIntrinsicsPipeline.from_pretrained(REPOS[variant], torch_dtype=dtype, **kw)
        except Exception:
            self.pipe = MarigoldIntrinsicsPipeline.from_pretrained(REPOS[variant], torch_dtype=dtype)
        self.pipe.set_progress_bar_config(disable=True)
        self.steps = steps
        self.props = dict(self.pipe.target_properties or {})
        self.names = list(self.props.get("target_names") or getattr(self.pipe, "target_names", []) or ["albedo"])
        self.dev = dev

    def to_gpu(self):
        self.pipe.to(self.dev)

    def to_cpu(self):
        self.pipe.to("cpu")
        free_vram()

    def infer(self, rgb_u8, seed=0):
        """-> {layer: (H, W, 3) or (H, W) float32}. 3-channel targets (albedo, shading...) are made
        linear; packed targets (appearance 'material' = roughness + metallicity) are split."""
        g = torch.Generator(device=self.dev).manual_seed(seed)        # same noise every frame: less flicker
        img = torch.from_numpy(rgb_u8).float().div(255).permute(2, 0, 1)[None]
        out = self.pipe(img, num_inference_steps=self.steps, generator=g, output_type="np")
        pred = out.prediction                                          # (numtargets, H, W, 3)
        res = {}
        for k, name in enumerate(self.names):
            a = np.asarray(pred[k], np.float32)
            p = self.props.get(name) or {}
            subs = p.get("sub_target_names") or []
            if any(subs):
                for c, sub in enumerate(subs[:3]):
                    if sub:
                        res[sub] = a[..., c]
            else:
                res[name] = srgb_to_linear(a) if p.get("prediction_space") == "srgb" else a
        return res
