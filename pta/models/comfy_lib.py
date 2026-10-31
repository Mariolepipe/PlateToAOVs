"""
ComfyUI's `comfy` package used as a plain Python library.

MoGe-3, LTX-2.5 (int8 'convrot' weights) and SAM 3.1 are only implemented in
ComfyUI's own code, so the app imports that code directly: no server, no web UI, no graph.
The node classes are just Python classes, called here like functions.
"""

import os
import sys

_READY = {}


def setup(log=print):
    if _READY:
        return _READY["root"]
    from .. import config
    root = config.PATHS["comfyui_dir"]
    if not os.path.isfile(os.path.join(root, "comfy", "model_management.py")):
        raise RuntimeError("ComfyUI code not found in %s (needed for MoGe-3, LTX and SAM 3.1). "
                           "Run setup.bat again, or set comfyui_dir in config.json." % root)
    if root not in sys.path:
        sys.path.insert(0, root)
    import comfy.options
    comfy.options.enable_args_parsing(False)          # do not read our command line
    from comfy.cli_args import args                   # defaults, like a plain launch
    import torch
    if not torch.cuda.is_available():
        args.cpu = True
    import folder_paths
    # make ComfyUI's model lookups see the same folders as the app
    kinds = ("diffusion_models", "unet", "loras", "text_encoders", "clip", "vae", "geometry_estimation",
             "checkpoints", "latent_upscale_models")
    for d in config.PATHS["model_dirs"]:
        for k in kinds:
            p = os.path.join(d, k)
            if os.path.isdir(p):
                target = "diffusion_models" if k == "unet" else ("text_encoders" if k == "clip" else k)
                try:
                    folder_paths.add_model_folder_path(target, p)
                except Exception:
                    pass
    _READY["root"] = root
    log("ComfyUI code loaded as a library from %s" % root)
    return root


def set_progress(fn):
    """fn(value, total) called by ComfyUI's samplers / VAE (None to remove)."""
    import comfy.utils
    comfy.utils.set_progress_bar_global_hook((lambda v, t, preview=None, node_id=None: fn(v, t)) if fn else None)


def out(node_output, i=0):
    """Value i of a node's result (new io.NodeOutput or old tuple)."""
    res = getattr(node_output, "result", node_output)
    return res[i]


def unload_all():
    import comfy.model_management as mm
    mm.unload_all_models()
    mm.soft_empty_cache()
