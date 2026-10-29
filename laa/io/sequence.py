"""
Image sequences: find the frames, read them, bring them to display-referred sRGB.

The models were trained on ordinary photos (display-referred sRGB, 0..1). A linear or ACEScg
EXR plate fed as is looks dark and washed out to them, so frames are converted first, exactly
like the Nuke panel did with the viewer transform.
"""

import os
import re

os.environ.setdefault("OPENCV_IO_ENABLE_OPENEXR", "1")   # must be set before cv2 is imported
import cv2  # noqa: E402
import numpy as np  # noqa: E402

IMAGE_EXT = (".exr", ".png", ".jpg", ".jpeg", ".tif", ".tiff", ".dpx")
_TOKEN = re.compile(r"(%0?(\d*)d|#+|\$F\d*)")
_NUM = re.compile(r"^(.*?)(\d+)(\.[A-Za-z0-9]+)$")

COLORSPACES = {
    "srgb": "sRGB / display (PNG, JPG, Nuke render)",
    "linear": "Linear Rec.709 / sRGB primaries (EXR)",
    "acescg": "ACEScg (EXR)",
}

# ACEScg (AP1) -> linear Rec.709, Bradford D60->D65 (same matrix as OCIO's ACES configs)
_AP1_TO_709 = np.array([[1.70505, -0.62179, -0.08326],
                        [-0.13026, 1.14080, -0.01055],
                        [-0.02400, -0.12897, 1.15297]], np.float32)


class Sequence(object):
    """frames: sorted list of (frame_number, path)."""

    def __init__(self, frames, name):
        self.frames = frames
        self.name = name

    @property
    def numbers(self):
        return [f for f, _ in self.frames]

    def __len__(self):
        return len(self.frames)

    def path(self, i):
        return self.frames[i][1]

    def subset(self, first=0, last=0):
        if not first and not last:
            return self
        fr = [(f, p) for f, p in self.frames if (not first or f >= first) and (not last or f <= last)]
        return Sequence(fr, self.name)


def find_sequence(spec):
    """spec: a folder, a pattern ('plate.%04d.exr', 'plate.####.exr') or one file of the sequence."""
    spec = spec.strip().strip('"').replace("\\", "/")
    if os.path.isdir(spec):
        files = sorted(f for f in os.listdir(spec) if f.lower().endswith(IMAGE_EXT))
        if not files:
            raise FileNotFoundError("no images in %s" % spec)
        groups = {}
        for f in files:
            m = _NUM.match(f)
            key = (m.group(1), m.group(3)) if m else (f, "")
            groups.setdefault(key, []).append(f)
        (pre, ext), names = max(groups.items(), key=lambda kv: len(kv[1]))
        frames = []
        for i, f in enumerate(names):
            m = _NUM.match(f)
            frames.append((int(m.group(2)) if m else i, os.path.join(spec, f)))
        return Sequence(sorted(frames), _clean_name(pre) or os.path.basename(spec.rstrip("/")))

    folder, base = os.path.split(spec)
    folder = folder or "."
    if _TOKEN.search(base):
        pre, post = _TOKEN.split(base, maxsplit=1)[0], _TOKEN.split(base, maxsplit=1)[-1]
    else:
        m = _NUM.match(base)
        if not m:
            if os.path.isfile(spec):
                return Sequence([(0, spec)], _clean_name(os.path.splitext(base)[0]))
            raise FileNotFoundError(spec)
        pre, post = m.group(1), m.group(3)
    rx = re.compile("^" + re.escape(pre) + r"(\d+)" + re.escape(post) + "$")
    frames = []
    for f in os.listdir(folder):
        m = rx.match(f)
        if m:
            frames.append((int(m.group(1)), os.path.join(folder, f)))
    if not frames:
        raise FileNotFoundError("no frames match %s" % spec)
    return Sequence(sorted(frames), _clean_name(pre))


def _clean_name(pre):
    return pre.strip("._- ") or "plate"


def read_float(path):
    """-> (H, W, 3) float32, values as stored in the file (8/16-bit scaled to 0..1)."""
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED | cv2.IMREAD_ANYDEPTH | cv2.IMREAD_ANYCOLOR)
    if img is None:
        raise IOError("cannot read %s" % path)
    if img.ndim == 2:
        img = img[..., None].repeat(3, -1)
    img = img[..., :3][..., ::-1]                      # BGR(A) -> RGB
    if img.dtype == np.uint8:
        img = img.astype(np.float32) / 255.0
    elif img.dtype == np.uint16:
        img = img.astype(np.float32) / 65535.0
    else:
        img = img.astype(np.float32)
    return np.ascontiguousarray(img)


def linear_to_srgb(x):
    x = np.clip(x, 0.0, None)
    return np.where(x <= 0.0031308, x * 12.92, 1.055 * np.power(x, 1 / 2.4) - 0.055)


def to_display(img, colorspace):
    """float image in `colorspace` -> display-referred sRGB 0..1."""
    if colorspace == "acescg":
        img = img @ _AP1_TO_709.T
        img = linear_to_srgb(img)
    elif colorspace == "linear":
        img = linear_to_srgb(img)
    return np.clip(img, 0.0, 1.0).astype(np.float32)


def read_display_u8(path, colorspace):
    """What the models see: 8-bit display-referred RGB (same as the PNGs Nuke rendered)."""
    return (to_display(read_float(path), colorspace) * 255.0 + 0.5).astype(np.uint8)


def write_png(path, img):
    """RGB (H,W,3) or single channel (H,W) uint8/uint16. Only colour images get RGB -> BGR:
    on a 2D array [..., ::-1] would flip the WIDTH (mirrored masks)."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if img.ndim == 3:
        img = img[..., ::-1]
    cv2.imwrite(path, np.ascontiguousarray(img), [cv2.IMWRITE_PNG_COMPRESSION, 1])


def read_png(path):
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise IOError("cannot read %s" % path)
    if img.ndim == 3:
        img = img[..., :3][..., ::-1]
    return np.ascontiguousarray(img)
