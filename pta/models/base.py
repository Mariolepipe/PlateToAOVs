"""Shared helpers for the model adapters: device, VRAM hygiene, Hugging Face downloads."""

import gc
import os

import torch


def device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def free_vram():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def hf_download(repo_id, filename, local_dir, log=print):
    """Download one file from Hugging Face into local_dir. If `filename` is not in the repo,
    take its only .safetensors / .pth file instead (repos sometimes rename files)."""
    from huggingface_hub import hf_hub_download, list_repo_files
    os.makedirs(local_dir, exist_ok=True)
    try:
        files = list_repo_files(repo_id)
    except Exception as e:
        raise RuntimeError("Hugging Face repo %s unreachable (%s). Download it by hand into %s"
                           % (repo_id, e, local_dir))
    if filename not in files:
        cand = [f for f in files if f.endswith((".safetensors", ".pth"))]
        if len(cand) != 1:
            raise RuntimeError("%s not found in %s (files: %s)" % (filename, repo_id, files))
        filename = cand[0]
    log("downloading %s/%s -> %s" % (repo_id, filename, local_dir))
    return hf_hub_download(repo_id=repo_id, filename=filename, local_dir=local_dir)
