"""Model downloads (Hugging Face). Files you already have (in any model folder of config.json,
e.g. your ComfyUI models) are found by name and skipped.

The app asks at start-up (Models window). From a terminal:
    python -m laa.download             the core models: MoGe-3, SAM 3.1, SAM 2.1, Sapiens2, depth, MatAnyone 2
    python -m laa.download moge|sam3|sam2|sapiens|depth|matanyone|ltx|cosmos   one of them
    python -m laa.download people|normals|depth|matanyone|alpha|albedo        what one feature needs
    python -m laa.download all         core + LTX (Foreground alpha, ~38 GB) + Cosmos (albedo, ~30 GB)

Gated repos (LTX-2.5, Cosmos, SAM 3.1): accept the license on their Hugging Face page first,
then enter a READ token when asked (huggingface.co > Settings > Access Tokens)."""
import os
import sys

from . import config

LTX = [  # (repo, file in the repo, ComfyUI models subfolder)
    ("Lightricks/LTX-2.5", "diffusion_models/ltx-2.5-22b-distilled-transformer-comfy-int8-convrot.safetensors", None),
    ("Lightricks/LTX-2.5", "text_encoders/gemma4-12b-with-proj-ltx-2.5-comfy-int8-convrot.safetensors", None),
    ("Lightricks/LTX-2.5", "vae/ltx-2.5-video-vae-conv-bf16.safetensors", None),
    ("Lightricks/LTX-2.5-22b-IC-LoRA-Alpha-Gen", "ltx-2.5-22b-ic-lora-alpha-gen-0.9.safetensors", "loras"),
]
MOGE = ("Comfy-Org/MoGe", "geometry_estimation/moge_3_vitg_fp16.safetensors", None)


def _root():
    return next((d for d in config.PATHS["model_dirs"] if os.path.isdir(d)), None) or config.default_download_dir("")


def _login():
    from huggingface_hub import get_token, login
    if not get_token():                        # the app passes HF_TOKEN; a terminal asks for it
        if not sys.stdin or not sys.stdin.isatty():
            raise RuntimeError("a Hugging Face token is needed for this model (gated)")
        print("This model is gated: accept its license on Hugging Face, then paste a READ token "
              "(huggingface.co > Settings > Access Tokens).")
        login()


def _get(repo, path, subdir=None, gated=False):
    name = os.path.basename(path)
    have = config.find_model(name)
    if have:
        print("  already there: %s" % have)
        return have
    if gated:
        _login()
    from huggingface_hub import hf_hub_download
    local = os.path.join(_root(), subdir) if subdir else _root()
    print("  downloading %s/%s" % (repo, path))
    return hf_hub_download(repo_id=repo, filename=path, local_dir=local)


def moge():
    _get(*MOGE)


def ltx():
    for repo, path, sub in LTX:
        _get(repo, path, sub, gated=True)


def sapiens():
    from .models.sapiens2 import checkpoint
    s = config.Settings()
    checkpoint("seg", s.seg_size)
    checkpoint("normal", s.sapiens_size)


def depth():
    from .models.vda import _REPOS
    s = config.Settings()
    if not config.find_model(s.vda_model, "videodepthanything"):
        from .models.base import hf_download
        hf_download(_REPOS[s.vda_model], s.vda_model, config.default_download_dir("videodepthanything"))
    else:
        print("  already there: %s" % s.vda_model)


def sam3():
    _login()
    from .models.sam3 import download
    download()


def sam2():
    from .models.sam2 import download
    download()


def matanyone():
    from .models.matanyone2 import download
    download()


def cosmos():
    from .models.cosmos_native import download
    download()


CORE = ["moge", "sam3", "sam2", "sapiens", "depth", "matanyone"]
ALL = {"moge": moge, "sam3": sam3, "sam2": sam2, "sapiens": sapiens, "depth": depth, "matanyone": matanyone,
       "ltx": ltx, "cosmos": cosmos}
# the groups of the app's Models window (laa.model_catalog)
GROUPS = {"people": ["sam3", "sam2"], "normals": ["sapiens", "moge"], "depth": ["depth"], "alpha": ["ltx"],
          "matanyone": ["matanyone"],
          "albedo": ["cosmos"]}


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    what = argv[0] if argv else "core"
    names = (CORE if what == "core" else CORE + ["ltx", "cosmos"] if what == "all" else
             [n for w in argv for n in GROUPS.get(w, [w])])
    bad = [n for n in names if n not in ALL]
    if bad:
        print("unknown: %s (expected: core, all, %s)" % (", ".join(bad), ", ".join(ALL)))
        return 1
    failed = []
    for n in names:
        print("\n== %s ==" % n)
        try:
            ALL[n]()
        except Exception as e:                 # keep going: one gated repo should not stop the rest
            print("  FAILED: %s" % e)
            failed.append(n)
    print("\nDone." if not failed else "\nDone, except: %s (see above, then run again)." % ", ".join(failed))
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
