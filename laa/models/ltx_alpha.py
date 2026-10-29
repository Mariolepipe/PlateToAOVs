"""
LTX-2.5 Alpha Gen (Lightricks IC-LoRA): soft video matte, hair and motion blur included.

Exactly the graph of the ComfyUI workflow, but the nodes are called as Python functions:
  UNET (int8) + Alpha Gen IC-LoRA, Gemma text encoder (empty prompt), LTX video VAE,
  plate added as IC-LoRA guide, 8 distilled steps (euler_ancestral, cfg 1), guides cropped,
  tiled decode. Plate fit/pad/8n+1 and the unpad/resize afterwards are the NukeBridge helpers.
"""

import torch
import torch.nn.functional as F

from . import comfy_lib

SIGMAS = "1.0, 0.99375, 0.9875, 0.98125, 0.975, 0.909375, 0.725, 0.421875, 0.0"


def prep(frames, max_w=1920, max_h=1088):
    """frames (B,H,W,3) float 0..1 torch -> padded (N,th,tw,3), info. Same as LTXAlphaPrep."""
    B, H, W, _ = frames.shape
    s = min(1.0, max_w / W, max_h / H)
    w, h = max(32, int(round(W * s))), max(32, int(round(H * s)))
    tw, th = (w + 31) // 32 * 32, (h + 31) // 32 * 32
    tw, th = min(tw, max_w), min(th, max_h)
    w, h = min(w, tw), min(h, th)
    x = frames[..., :3].permute(0, 3, 1, 2)
    if (h, w) != (H, W):
        x = F.interpolate(x, size=(h, w), mode="bilinear", align_corners=False, antialias=True)
    px, py = (tw - w) // 2, (th - h) // 2
    x = F.pad(x, (px, tw - w - px, py, th - h - py), mode="replicate")
    n = ((B - 1 + 7) // 8) * 8 + 1
    if n > B:
        x = torch.cat([x, x[-1:].expand(n - B, -1, -1, -1)], 0)
    return x.permute(0, 2, 3, 1).contiguous(), {"orig": (B, H, W), "box": (px, py, w, h), "size": (tw, th, n)}


def finish(matte, info):
    """decoded matte (N,th,tw,3) -> (B,H,W) float 0..1. Same as LTXAlphaFinish."""
    B, H, W = info["orig"]
    px, py, w, h = info["box"]
    m = matte[..., :3].float().mean(-1)[:B, py:py + h, px:px + w]
    if m.shape[0] < B:
        m = torch.cat([m, m[-1:].expand(B - m.shape[0], -1, -1)], 0)
    if (h, w) != (H, W):
        m = F.interpolate(m[:, None], size=(H, W), mode="bilinear", align_corners=False)[:, 0]
    return m.clamp(0, 1).cpu().numpy()


class LTXAlpha(object):
    """say(text) reports what is happening (loading, step 3/8...), for the status line."""

    # ComfyUI runs every node inside torch.inference_mode(); called outside it, the VAE decode
    # fails ("Inplace update to inference tensor outside InferenceMode"). Same mode here.
    def __init__(self, s, log=print, say=None):
        with torch.inference_mode():
            self._init(s, log, say)

    def matte(self, frames_u8, progress=None, label=""):
        with torch.inference_mode():
            return self._matte(frames_u8, progress, label)

    def _init(self, s, log=print, say=None):
        comfy_lib.setup(log)
        import gc
        import nodes
        from comfy_extras import nodes_lt
        self.nodes, self.lt = nodes, nodes_lt
        self.say = say or (lambda t: None)
        out = comfy_lib.out
        # text encoder first, then dropped before the 22B transformer is loaded: never both in memory
        self.say("Loading the text encoder (1/3)")
        log("loading Gemma text encoder %s" % s.ltx_text_encoder)
        clip = out(nodes.CLIPLoader().load_clip(s.ltx_text_encoder, "ltxv", "default"))
        self.say("Encoding the prompt")
        cond = out(nodes.CLIPTextEncode().encode(clip, ""))
        self.pos, self.neg = (comfy_lib.out(nodes_lt.LTXVConditioning.execute(positive=cond, negative=cond,
                                                                              frame_rate=float(s.ltx_fps)), k)
                              for k in (0, 1))
        del clip, cond
        comfy_lib.unload_all()
        gc.collect()
        self.say("Loading LTX-2.5 (2/3)")
        log("loading LTX-2.5 %s + Alpha Gen IC-LoRA" % s.ltx_unet)
        model = out(nodes.UNETLoader().load_unet(s.ltx_unet, "default"))
        self.model = out(nodes.LoraLoaderModelOnly().load_lora_model_only(model, s.ltx_lora, 1.0))
        self.iclora = out(nodes_lt.GetICLoRAParameters.execute(iclora_model=self.model))
        self.say("Loading the video VAE (3/3)")
        self.vae = out(nodes.VAELoader().load_vae(s.ltx_vae))
        self.s = s

    def _matte(self, frames_u8, progress=None, label=""):
        """frames_u8 (B,H,W,3) uint8 -> (B,H,W) float alpha. progress(fraction 0..1 of this chunk)."""
        from comfy_extras import nodes_custom_sampler as cs
        out, lt, s = comfy_lib.out, self.lt, self.s
        x = torch.from_numpy(frames_u8).float() / 255.0
        img, info = prep(x, s.ltx_max_width, s.ltx_max_height)
        tw, th, n = info["size"]
        self.say("Preparing %s" % label)
        latent = out(lt.EmptyLTXVLatentVideo.execute(width=tw, height=th, length=n, batch_size=1))
        g = lt.LTXVAddGuide.execute(positive=self.pos, negative=self.neg, vae=self.vae, latent=latent, image=img,
                                    frame_idx=0, strength=1.0, iclora_parameters=self.iclora)
        pos, neg, lat = out(g, 0), out(g, 1), out(g, 2)
        noise = out(cs.RandomNoise.execute(noise_seed=int(s.ltx_seed)))
        sampler = out(cs.KSamplerSelect.execute(sampler_name="euler_ancestral"))
        sigmas = out(cs.ManualSigmas.execute(sigmas=SIGMAS))
        guider = out(cs.CFGGuider.execute(model=self.model, positive=pos, negative=neg, cfg=1.0))
        prog = progress or (lambda f: None)

        def on_sample(v, t):
            self.say("Step %d of %d  %s" % (v, t, label))
            prog(0.9 * v / max(t, 1))

        def on_decode(v, t):
            self.say("Decoding %s" % label)
            prog(0.9 + 0.1 * v / max(t, 1))

        self.say("Step 0 of 8  %s" % label)
        comfy_lib.set_progress(on_sample)
        try:
            samples = out(cs.SamplerCustomAdvanced.execute(noise=noise, guider=guider, sampler=sampler,
                                                            sigmas=sigmas, latent_image=lat), 0)
            c = lt.LTXVCropGuides.execute(positive=pos, negative=neg, latent=samples)
            self.say("Decoding %s" % label)
            comfy_lib.set_progress(on_decode)
            decoded = out(self.nodes.VAEDecodeTiled().decode(self.vae, out(c, 2), 512, 64, 64, 16))
        finally:
            comfy_lib.set_progress(None)
        return finish(decoded, info)

    def release(self):
        with torch.inference_mode():                 # weights were loaded in inference mode too
            self.model = self.vae = None
            comfy_lib.unload_all()


def chunks_with_overlap(n, chunk, overlap):
    """[(lo, hi, keep_lo, keep_hi)]: process lo..hi, keep keep_lo..keep_hi (absolute indices)."""
    out = []
    for c0 in range(0, n, chunk):
        c1 = min(n, c0 + chunk)
        out.append((max(0, c0 - overlap), min(n, c1 + overlap), c0, c1))
    return out

