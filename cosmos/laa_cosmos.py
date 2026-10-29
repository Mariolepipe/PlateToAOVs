"""
LiveActionAOVs <-> Cosmos DiffusionRenderer (inverse renderer, NVIDIA). Runs INSIDE WSL, in the
cosmos-predict1 conda env, from the cosmos1-diffusion-renderer repo folder:

  python laa_cosmos.py --in_dir /mnt/d/.../cosmos_in --out_dir /mnt/d/.../cosmos_out --passes basecolor

in_dir : 00000.png, 00001.png ... frames already letterboxed to 1280x704 by LAA
out_dir: <pass>/00000.png ...  (uint8, what the model outputs), overlaps between the 57-frame
         chunks are cross-faded so there is no pop at the chunk boundary.
Progress: lines "@@COSMOS {json}" on stdout.
"""

import argparse
import json
import os
import sys
import time

import numpy as np


def send(**m):
    sys.stdout.write("@@COSMOS " + json.dumps(m) + "\n")
    sys.stdout.flush()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_dir", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--passes", nargs="+", default=["basecolor"])
    ap.add_argument("--checkpoint_dir", default="checkpoints")
    ap.add_argument("--frames", type=int, default=57)
    ap.add_argument("--overlap", type=int, default=8)
    ap.add_argument("--steps", type=int, default=15)
    ap.add_argument("--seed", type=int, default=1000)
    ap.add_argument("--offload", type=int, default=1)
    a = ap.parse_args()

    import torch
    from PIL import Image
    from torch.utils.data import DataLoader
    from cosmos_predict1.diffusion.inference.diffusion_renderer_pipeline import DiffusionRendererPipeline
    from cosmos_predict1.diffusion.inference.diffusion_renderer_utils.rendering_utils import GBUFFER_INDEX_MAPPING
    from cosmos_predict1.diffusion.inference.diffusion_renderer_utils.dataset_inference import VideoFramesDataset
    from cosmos_predict1.diffusion.inference.diffusion_renderer_utils.dataloader_utils import dict_collation_fn
    torch.enable_grad(False)

    ds = VideoFramesDataset(root_dir=a.in_dir, sample_n_frames=a.frames, overlap_n_frames=a.overlap,
                            chunk_mode="all", group_mode="folder", image_extensions=[".png"],
                            resolution=[704, 1280])
    chunks = [[int(os.path.splitext(os.path.basename(p))[0]) for p in c] for c in ds.chunks]
    n_total = max(max(c) for c in chunks) + 1
    send(t="status", m="Loading Cosmos DiffusionRenderer (7B)", chunks=len(chunks), frames=n_total)
    t0 = time.time()
    pipe = DiffusionRendererPipeline(
        checkpoint_dir=a.checkpoint_dir, checkpoint_name="Diffusion_Renderer_Inverse_Cosmos_7B",
        offload_network=bool(a.offload), offload_tokenizer=bool(a.offload),
        offload_text_encoder_model=False, offload_guardrail_models=False,
        guidance=0.0, num_steps=a.steps, height=704, width=1280, fps=24,
        num_video_frames=a.frames, seed=a.seed)
    send(t="log", m="Cosmos loaded in %.0fs" % (time.time() - t0))

    acc = {p: {} for p in a.passes}            # frame -> [sum float32, weight]
    loader = DataLoader(ds, batch_size=1, shuffle=False, num_workers=0, collate_fn=dict_collation_fn)
    units, done = len(chunks) * len(a.passes), 0
    for ci, batch in enumerate(loader):
        idxs = chunks[ci]
        nxt = chunks[ci + 1][0] if ci + 1 < len(chunks) else None
        for p in a.passes:
            send(t="status", m="%s, frames %d-%d" % (p, idxs[0], idxs[-1]), done=done, total=units)
            batch["context_index"].fill_(GBUFFER_INDEX_MAPPING[p])
            out = pipe.generate_video(data_batch=batch, normalize_normal=(p == "normal"))   # (T,H,W,3) uint8
            seen = set()
            for k, fi in enumerate(idxs):
                if fi in seen:                    # last chunk is padded by repeating its last frame
                    continue
                seen.add(fi)
                w = 1.0
                if ci > 0 and k < a.overlap:      # fade in over the overlap with the previous chunk
                    w = min(w, (k + 1.0) / (a.overlap + 1.0))
                if nxt is not None and fi >= nxt:  # fade out where the next chunk starts
                    w = min(w, (len(idxs) - k) / (a.overlap + 1.0))
                cur = acc[p].setdefault(fi, [np.zeros(out.shape[1:], np.float32), 0.0])
                cur[0] += out[k].astype(np.float32) * w
                cur[1] += w
            # frames no later chunk touches are final: write them
            os.makedirs(os.path.join(a.out_dir, p), exist_ok=True)
            for fi in sorted(acc[p]):
                if nxt is None or fi < nxt:
                    s, wsum = acc[p].pop(fi)
                    img = np.clip(s / max(wsum, 1e-6) + 0.5, 0, 255).astype(np.uint8)
                    Image.fromarray(img).save(os.path.join(a.out_dir, p, "%05d.png" % fi), compress_level=1)
            done += 1
            send(t="progress", done=done, total=units)
    send(t="finished", seconds=time.time() - t0)


if __name__ == "__main__":
    main()
