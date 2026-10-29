"""laa_cosmos.py chunking / cross-fade with a fake Cosmos pipeline (needs the cosmos repo for its dataset code)."""
import sys, os, types, tempfile, numpy as np, runpy
REPO = os.environ.get("COSMOS_REPO", os.path.expanduser("~/cosmos/cosmos1-diffusion-renderer"))
sys.path.insert(0, REPO)
import torch
calls = []
class FakePipe:
    def __init__(self, **k): self.k = k
    def generate_video(self, data_batch, normalize_normal=False):
        v = data_batch["rgb"][0]                     # (C,N,H,W) -1..1
        calls.append((int(data_batch["context_index"][0]), v.shape))
        return (((v + 1) / 2).permute(1, 2, 3, 0) * 255).round().to(torch.uint8).numpy()
mod = types.ModuleType("cosmos_predict1.diffusion.inference.diffusion_renderer_pipeline")
mod.DiffusionRendererPipeline = FakePipe
sys.modules[mod.__name__] = mod
from PIL import Image
d = tempfile.mkdtemp(); din, dout = d + "/in", d + "/out"; os.makedirs(din)
N = 130
for i in range(N):
    Image.fromarray(np.full((704, 1280, 3), i, np.uint8)).save(din + "/%05d.png" % i)
sys.argv = ["laa_cosmos.py", "--in_dir", din, "--out_dir", dout, "--passes", "basecolor", "roughness", "--overlap", "8"]
runpy.run_path(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "cosmos", "laa_cosmos.py"), run_name="__main__")
for p in ("basecolor", "roughness"):
    files = sorted(os.listdir(dout + "/" + p)); assert len(files) == N, (p, len(files))
    vals = [int(np.asarray(Image.open(dout + "/" + p + "/" + f))[0, 0, 0]) for f in files]
    assert vals == list(range(N)), vals[:70]
print("calls:", len(calls), calls[0])
print("OK")
