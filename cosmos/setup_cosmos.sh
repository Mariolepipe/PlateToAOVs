#!/usr/bin/env bash
# Installs Cosmos DiffusionRenderer (inverse renderer only) inside WSL, for LiveActionAOVs.
# Run by setup_cosmos.bat. Safe to run again: every step is skipped when already done.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$HOME/cosmos"
REPO="$ROOT/cosmos1-diffusion-renderer"
CONDA="$HOME/miniforge3"
say() { printf '\n\033[1;34m== %s\033[0m\n' "$*"; }

say "1/6 System packages (asks for your Linux password)"
sudo apt-get update -y
sudo apt-get install -y git wget build-essential libgl1 libglib2.0-0 ffmpeg

say "2/6 Miniforge (conda)"
if [ ! -x "$CONDA/bin/conda" ]; then
  wget -q --show-progress -O /tmp/miniforge.sh \
    "https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh"
  bash /tmp/miniforge.sh -b -p "$CONDA"
fi
source "$CONDA/etc/profile.d/conda.sh"

say "3/6 Cosmos DiffusionRenderer code"
mkdir -p "$ROOT"
if [ ! -d "$REPO/.git" ]; then
  git clone --depth 1 https://github.com/nv-tlabs/cosmos1-diffusion-renderer.git "$REPO"
fi
cd "$REPO"

say "4/6 Python environment (cosmos-predict1). Long: CUDA toolkit + PyTorch + Transformer Engine"
if ! conda env list | grep -q "^cosmos-predict1 "; then
  conda env create --file cosmos-predict1.yaml
fi
conda activate cosmos-predict1
pip install -r requirements.txt
# NVIDIA's documented patch for Transformer Engine in conda envs
ln -sf "$CONDA_PREFIX"/lib/python3.10/site-packages/nvidia/*/include/* "$CONDA_PREFIX/include/" || true
ln -sf "$CONDA_PREFIX"/lib/python3.10/site-packages/nvidia/*/include/* "$CONDA_PREFIX/include/python3.10" || true
if ! python -c "import transformer_engine" 2>/dev/null; then
  pip install "transformer-engine[pytorch]==1.12.0"
fi
python -c "import torch; assert torch.cuda.is_available(), 'CUDA not visible in WSL'; print('GPU:', torch.cuda.get_device_name(0))"

say "5/6 Model weights (inverse renderer + tokenizer only, ~30 GB)"
if ! python -c "from huggingface_hub import HfApi; HfApi().whoami()" >/dev/null 2>&1; then
  echo "Paste a Hugging Face token with READ access (huggingface.co > Settings > Access Tokens)."
  huggingface-cli login
fi
CUDA_HOME="$CONDA_PREFIX" PYTHONPATH="$(pwd)" python - <<'PY'
from huggingface_hub import snapshot_download
pat = ["README.md", "model.pt", "mean_std.pt", "image_mean_std.pt", "config.json", "*.jit", "guardrail/*"]
for name in ("Diffusion_Renderer_Inverse_Cosmos_7B", "Cosmos-Tokenize1-CV8x8x8-720p"):
    print("downloading nvidia/" + name)
    snapshot_download(repo_id="nvidia/" + name, local_dir="checkpoints/" + name, allow_patterns=pat)
PY

say "6/6 Test on one example image"
mkdir -p /tmp/laa_cosmos_test/in /tmp/laa_cosmos_test/out
python - <<'PY'
import glob
from PIL import Image
src = sorted(glob.glob("asset/examples/image_examples/*"))[0]
Image.open(src).convert("RGB").resize((1280, 704)).save("/tmp/laa_cosmos_test/in/00000.png")
print("test image:", src)
PY
CUDA_HOME="$CONDA_PREFIX" PYTHONPATH="$(pwd)" python "$HERE/laa_cosmos.py" \
  --in_dir /tmp/laa_cosmos_test/in --out_dir /tmp/laa_cosmos_test/out --passes basecolor --frames 1 --overlap 0
ls -la /tmp/laa_cosmos_test/out/basecolor
say "Cosmos is ready. In LiveActionAOVs: Albedo > Best (Cosmos)."
