"""
Paths (config.json next to the app) and the job settings (one dataclass, saved with each job).

config.json only says where things live on this machine. Model files are found by file name
in any of `model_dirs` (searched recursively), so the ComfyUI folders can be reused as they are.
"""

import dataclasses
import json
import os
from pathlib import Path

APP_DIR = Path(__file__).resolve().parents[1]
CONFIG_PATH = APP_DIR / "config.json"

DEFAULT_PATHS = {
    # ComfyUI's code, used as a library for MoGe-3, LTX-2.5 and SAM 3.1 (no server, no web UI).
    # setup.bat finds your ComfyUI or downloads one into vendor/ComfyUI and writes config.json.
    "comfyui_dir": "vendor/ComfyUI",
    # model files are found by file name in any of these folders (searched recursively), so
    # ComfyUI's models folders can be reused as they are. "models" = the app's own folder.
    "model_dirs": ["models"],
    # official code of the models (copied into vendor/ by setup.bat)
    "sapiens2_repo": "vendor/sapiens2",
    "vda_code": "vendor/vda",
}
COMFYUI_TAG = "v0.39.2"          # the ComfyUI version the app is tested with (downloaded if you have none)


def _abs(p):
    p = os.path.expanduser(os.path.expandvars(str(p)))
    return str(Path(p) if os.path.isabs(p) else (APP_DIR / p))


def load_paths():
    data = dict(DEFAULT_PATHS)
    if CONFIG_PATH.is_file():
        with open(CONFIG_PATH, "r", encoding="utf-8") as f:
            data.update(json.load(f))
    data["comfyui_dir"] = _abs(data["comfyui_dir"])
    data["model_dirs"] = [_abs(d) for d in data["model_dirs"]]
    data["sapiens2_repo"] = _abs(data["sapiens2_repo"])
    data["vda_code"] = _abs(data["vda_code"])
    return data


PATHS = load_paths()
_FOUND = {}


def find_model(filename, subdir_hint=None):
    """Full path of a model file, searched by name in every model_dir. None if missing."""
    key = (filename, subdir_hint)
    if key in _FOUND:
        return _FOUND[key]
    for root in PATHS["model_dirs"]:
        if not os.path.isdir(root):
            continue
        if subdir_hint:
            cand = os.path.join(root, subdir_hint, filename)
            if os.path.isfile(cand):
                _FOUND[key] = cand
                return cand
        for dirpath, dirnames, filenames in os.walk(root):
            dirnames[:] = [d for d in dirnames if not d.startswith(".")]
            if filename in filenames:
                _FOUND[key] = os.path.join(dirpath, filename)
                return _FOUND[key]
    return None


def require_model(filename, subdir_hint=None, hint=""):
    p = find_model(filename, subdir_hint)
    if p is None:
        raise FileNotFoundError("Model file not found: %s (searched %s). %s"
                                % (filename, ", ".join(PATHS["model_dirs"]), hint))
    return p


def default_download_dir(subdir):
    """Where a missing model gets downloaded: the first existing model_dir."""
    for root in PATHS["model_dirs"]:
        if os.path.isdir(root):
            d = os.path.join(root, subdir)
            os.makedirs(d, exist_ok=True)
            return d
    d = os.path.join(_abs("models"), subdir)
    os.makedirs(d, exist_ok=True)
    return d


@dataclasses.dataclass
class Settings:
    # ---- input ----
    input_pattern: str = ""            # e.g. C:/shots/plate.%04d.exr  or a folder
    first: int = 0                     # 0/0 = whole sequence
    last: int = 0
    colorspace: str = "srgb"           # srgb | linear | acescg   (what the input frames are in)
    output_dir: str = ""
    name: str = ""                     # output file prefix, default = input name

    # ---- which passes to make ----
    normals_enabled: bool = True       # N (Sapiens2 on people + MoGe elsewhere)
    alpha_enabled: bool = True
    # depth_enabled (depth.Z + P) and albedo_enabled are below with their models

    # ---- people: who is who (SAM 3.1, numbered, checked by you before the rest runs) ----
    people_enabled: bool = True        # off: people are told apart by touching blobs of the Sapiens seg
    people_review: bool = False        # Generate stops after finding people (the app uses its People page instead)
    sam3_model: str = "sam3.1_multiplex_fp16.safetensors"
    people_prompt: str = "person"
    people_threshold: float = 0.5      # detection confidence
    people_max: int = 16               # most people followed at once
    people_detect_interval: int = 1    # look for new people every N frames

    # ---- people: segmentation + normal crops (Sapiens2) ----
    sapiens_size: str = "1b"           # 0.4b 0.8b 1b 5b (normal model)
    seg_size: str = "1b"
    crop_padding: float = 0.08
    head_pass: bool = True
    head_fraction: float = 0.3
    feather_px: int = 24
    mask_erode_px: int = 3
    mask_blur_px: int = 6
    crop_chunk: int = 25               # crop boxes are fixed over this many frames

    # ---- background normals / camera (MoGe) ----
    moge_model: str = "moge_3_vitg_fp16.safetensors"
    moge_resolution_level: int = 9
    fov_x: float = 86.0                # 0 = auto
    moge_refine_steps: int = 3

    # ---- depth (Video Depth Anything, metric) ----
    depth_enabled: bool = True         # depth.Z (Video Depth Anything metric) + P
    vda_model: str = "metric_video_depth_anything_vitl.pth"
    vda_input_size: int = 518
    vda_max_res: int = 1280
    vda_fp16: bool = True

    # ---- temporal stabilization of N (RAFT optical flow) ----
    stab_radius: int = 2
    stab_sigma: float = 0.04
    stab_max_angle: float = 15.0

    # ---- alpha ----
    alpha_source: str = "matanyone"    # matanyone (Selection: MatAnyone 2 per number) | ltx (Foreground,
                                       # LTX on the whole frame) | matting (Sapiens2 matting) | sapiens (seg, hard)
    matanyone_max_side: int = 720      # MatAnyone 2 works at this shorter side. Tested on a 2K shot: at
                                       # 1080 it starts matting the person next to yours, at 720 it does not
    matanyone_warmup: int = 10         # first-frame refinement passes
    matanyone_grow: int = 10           # px: closes small holes of the SAM mask before matting
    ltx_unet: str = "ltx-2.5-22b-distilled-transformer-comfy-int8-convrot.safetensors"
    ltx_lora: str = "ltx-2.5-22b-ic-lora-alpha-gen-0.9.safetensors"
    ltx_text_encoder: str = "gemma4-12b-with-proj-ltx-2.5-comfy-int8-convrot.safetensors"
    ltx_vae: str = "ltx-2.5-video-vae-conv-bf16.safetensors"
    ltx_max_width: int = 1920
    ltx_max_height: int = 1088
    ltx_seed: int = 1234
    ltx_chunk: int = 25
    ltx_overlap: int = 2
    ltx_fps: float = 24.0

    # ---- albedo ----
    albedo_enabled: bool = False
    albedo_source: str = "cosmos"      # cosmos (NVIDIA DiffusionRenderer) | marigold
    cosmos_backend: str = "native"     # native (Windows, Materia port) | wsl (official NVIDIA code in WSL)
    cosmos_guidance: float = 2.0       # tested on shot030: 2 keeps dark materials dark, 0 washes the set out
    cosmos_material: bool = False      # + roughness and metallic (each pass = one more full run)
    cosmos_geometry: bool = False      # + Cosmos normal and depth, as comparison layers
    cosmos_steps: int = 15
    cosmos_face_crops: bool = False    # Cosmos again on a crop per head (off: it hallucinated blotchy skin)
    cosmos_face_size: float = 0.5      # head height / crop height (smaller = more context, less detail)
    cosmos_face_edge: float = 0.012    # width of the blend inside the head edge, fraction of the crop height
    cosmos_overlap: int = 8            # frames shared by two 57-frame chunks, cross-faded
    cosmos_offload: bool = True        # needed on 24 GB cards (peak is ~27 GB without)
    wsl_distro: str = "Ubuntu-22.04"
    albedo_variant: str = "appearance"  # Marigold: appearance (albedo, roughness, metal) | lighting (albedo, shading)
    albedo_steps: int = 4

    # ---- output ----
    half_float: bool = True
    comparison_layers: bool = False    # Nraw, Nmoge, depth_moge.Z, moge_valid.A
    previews: bool = True              # PNG per step for the viewer

    def to_json(self):
        return json.dumps(dataclasses.asdict(self), indent=2)

    @classmethod
    def from_dict(cls, d):
        names = {f.name for f in dataclasses.fields(cls)}
        return cls(**{k: v for k, v in d.items() if k in names})
