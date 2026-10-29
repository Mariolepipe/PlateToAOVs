"""
The job: plate -> people -> seg -> crops -> moge -> merge -> stabilize -> depth -> alpha -> albedo -> write.
(normals first: the main pass is ready to look at before the slow alpha starts)

Each step runs over the whole sequence with one model in VRAM, writes its result per frame in
<output>/_cache/<step>/ and a preview per frame in <output>/_preview/<view>/. A step is skipped
when its inputs and settings have not changed since the last run (hash in _done.json), so
changing the crop padding only re-runs crops -> merge -> stabilize -> write, not MoGe or LTX.
"""

import dataclasses
import hashlib
import json
import math
import os
import shutil
import time

import numpy as np
import torch

from . import config
from .io import sequence as seq
from .io.exr_writer import write_exr, HALF, FLOAT
from .stages import crops as C
from .stages import debug as D
from .stages import temporal
from .models.base import free_vram

GPU_STEPS = ("people", "seg", "crops", "moge", "stabilize", "depth", "alpha", "albedo")


def vram_text():
    """'5.1 / 24.0 GB free' (what every process on the card leaves us)."""
    try:
        if not torch.cuda.is_available():
            return "no GPU"
        free, total = torch.cuda.mem_get_info()
        return "%.1f / %.1f GB free" % (free / 2 ** 30, total / 2 ** 30)
    except Exception:
        return "unknown"

# step -> (settings it depends on, upstream steps)
STEPS = [
    ("plate", ("input_pattern", "first", "last", "colorspace"), ()),
    ("people", ("people_enabled", "sam3_model", "people_prompt", "people_threshold", "people_max",
                "people_detect_interval"), ("plate",)),
    ("seg", ("seg_size",), ("plate",)),
    ("crops", ("normals_enabled", "sapiens_size", "crop_padding", "head_pass", "head_fraction", "feather_px",
               "crop_chunk"), ("seg",)),
    ("moge", ("moge_model", "moge_resolution_level", "fov_x", "moge_refine_steps"), ("plate",)),
    ("merge", ("mask_erode_px", "mask_blur_px"), ("crops", "moge", "seg")),
    ("stabilize", ("stab_radius", "stab_sigma", "stab_max_angle"), ("merge",)),
    ("depth", ("vda_model", "vda_input_size", "vda_max_res", "vda_fp16"), ("plate",)),
    ("alpha", ("alpha_source", "ltx_unet", "ltx_lora", "ltx_text_encoder", "ltx_vae", "ltx_max_width",
               "ltx_max_height", "ltx_seed", "ltx_chunk", "ltx_overlap", "ltx_fps", "feather_px",
               "matanyone_max_side", "matanyone_warmup", "matanyone_grow"), ("seg", "crops")),
    ("albedo", ("albedo_source", "albedo_variant", "albedo_steps", "cosmos_material", "cosmos_geometry",
                "cosmos_steps", "cosmos_overlap", "cosmos_backend", "cosmos_guidance", "cosmos_face_crops",
                "cosmos_face_size", "cosmos_face_edge"), ("plate",)),
    ("write", ("half_float", "comparison_layers", "name", "output_dir", "normals_enabled", "depth_enabled",
               "alpha_enabled", "albedo_enabled"), ("stabilize", "moge", "depth", "alpha", "albedo")),
]
STEP_NAMES = [s[0] for s in STEPS]

# bump when a step's code changes its output: its cache (and everything after it) is redone
STEP_VERSION = {"people": 4,              # 4: SAM 2.1 keyframes + propagation (Sammie-Roto style)
                "seg": 2,                 # 2: single-channel PNGs were written mirrored
                "albedo": 3,              # 2: face crops on the head only, 3: blend inside the head
                "alpha": 8,               # 4: one crop per numbered person (SAM3), 5: objects = SAM mask
                                          # 6: what a person holds stays in their alpha again
                                          # 7: no bleed from people you did not select
                                          # 8: Foreground (LTX, whole frame) / Selection (MatAnyone 2)
                "crops": 3,               # 2: one crop per numbered person, 3: the crop holds all of them
                "write": 3}               # 2: passes can be turned off, 3: people id layer


class Cancelled(Exception):
    pass


def _hash(obj):
    return hashlib.sha1(json.dumps(obj, sort_keys=True, default=str).encode()).hexdigest()[:16]


class Job(object):
    def __init__(self, settings, log=print, progress=None, cancel=None, detail=None):
        self.s = settings
        self.log = log
        self.detail = detail or (lambda text: None)    # one-line "what is happening now" for the UI
        self._progress = progress or (lambda step, d, t: None)
        self._cancel = cancel or (lambda: False)
        src = seq.find_sequence(settings.input_pattern)
        self.full_range = (src.numbers[0], src.numbers[-1]) if len(src) else (0, 0)   # the whole plate
        self.seq = src.subset(settings.first, settings.last)
        if not len(self.seq):
            raise ValueError("no frames in the selected range")
        self.name = settings.name or self.seq.name
        self.out = settings.output_dir or os.path.join(os.path.dirname(self.seq.path(0)), "aovs_" + self.name)
        self.cache = os.path.join(self.out, "_cache")
        self.prev = os.path.join(self.out, "_preview")
        os.makedirs(self.cache, exist_ok=True)
        self.n = len(self.seq)
        self._ran = set()

    # ------------------------------------------------------------------ helpers
    def frame_label(self, i):
        return "%04d" % self.seq.numbers[i]

    def cpath(self, step, i, ext):
        return os.path.join(self.cache, step, "%05d.%s" % (i, ext))

    def ppath(self, view, i):
        return os.path.join(self.prev, view, "%05d.jpg" % i)

    def preview(self, view, i, rgb_u8):
        if not self.s.previews:
            return
        p = self.ppath(view, i)
        os.makedirs(os.path.dirname(p), exist_ok=True)
        import cv2
        img = rgb_u8
        if img.shape[1] > 2048:
            sc = 2048.0 / img.shape[1]
            img = cv2.resize(img, (2048, int(img.shape[0] * sc)), interpolation=cv2.INTER_AREA)
        cv2.imwrite(p, np.ascontiguousarray(img[..., ::-1]), [cv2.IMWRITE_JPEG_QUALITY, 92])

    def check(self):
        if self._cancel():
            raise Cancelled()

    def progress(self, step, d, t):
        self._progress(step, d, t)

    def plate(self, i):
        return seq.read_png(self.cpath("plate", i, "png"))

    def seg(self, i):
        return seq.read_png(self.cpath("seg", i, "png"))

    # ------------------------------------------------------------------ people (SAM3 numbers)
    def people_on(self):
        return bool(self.s.people_enabled) and self.needed("people")

    def edits_path(self):
        return os.path.join(self.out, "laa_people.json")

    def _people_base(self, legacy=False):
        """what your clicks and names belong to: the plate (a new plate = nothing to keep). What
        Find everyone looks for is not part of it: it starts the selection again by itself."""
        deps = ("people_enabled", "sam3_model") + (("people_prompt", "people_threshold", "people_max",
                                                     "people_detect_interval") if legacy else ())
        return _hash({"plate": self._read_done("plate").get("run"),
                      "deps": {k: getattr(self.s, k, None) for k in deps}})

    def people_edits(self):
        from .stages import people as PP
        base = self._people_base()
        try:
            with open(self.edits_path()) as f:
                e = json.load(f)
        except Exception:
            e = None
        if isinstance(e, dict) and e.get("base") == self._people_base(legacy=True):
            e["base"] = base                      # saved by an older version: same plate, keep it
        if not isinstance(e, dict) or e.get("base") != base:
            return PP.empty_edits(base)
        out = PP.empty_edits(base)
        out.update(e)
        return out

    def clear_people(self):
        """start the selection again: no clicks, names or objects, nothing found yet"""
        from .stages import people as PP
        self.save_people_edits(PP.empty_edits(self._people_base()))
        for d in (os.path.join(self.cache, "people"), os.path.join(self.prev, "people"),
                  os.path.join(self.out, "laa_people_keys")):
            shutil.rmtree(d, ignore_errors=True)

    def save_people_edits(self, e):
        e = dict(e)
        e["base"] = self._people_base()
        with open(self.edits_path(), "w") as f:
            json.dump(e, f, indent=1)

    def people_ids(self, i, final=True):
        """(h, w) uint8 at the working size: 0 nobody, k person k (after your review if final)"""
        ids = seq.read_png(self.cpath("people", i, "png"))
        if final:
            from .stages import people as PP
            if not hasattr(self, "_lut") or self._lut[0] != self._edits_key():
                self._lut = (self._edits_key(), PP.lut(self.people_edits()))
            ids = self._lut[1][ids]
        return ids

    def _edits_key(self):
        try:
            return os.path.getmtime(self.edits_path())
        except OSError:
            return 0

    def people_ids_full(self, i, H, W):
        import cv2
        ids = self.people_ids(i)
        if ids.shape != (H, W):
            ids = cv2.resize(ids, (W, H), interpolation=cv2.INTER_NEAREST)
        return ids

    def object_ids(self):
        """numbers you added as objects (a car, a prop...): alpha and people.id, no Sapiens crops"""
        if not self.people_on():
            return set()
        kinds = dict(self._read_done("people").get("kinds", {}))     # found by SAM 3.1 as "car", "house"...
        kinds.update(self.people_edits().get("kinds", {}))           # what you added / changed
        return {int(k) for k, v in kinds.items() if v == "object"}

    def person(self, i, objects=False, with_objects=True):
        """bool (H, W): people pixels. Sapiens2 edge, but only where a kept person (SAM) is.
        objects=True also keeps everything inside each person's SAM mask (the rifle you clicked as
        part of them, a bag...): for the alpha, not for the normals (Sapiens only knows bodies).
        with_objects=False leaves out the numbers marked as objects (they get their own alpha)."""
        if self.only_objects():                   # no Sapiens: no people pixels, objects = their SAM mask
            H, W = self.plate(i).shape[:2]
            full = self.people_ids_full(i, H, W)
            return full > 0 if objects and with_objects else np.zeros((H, W), bool)
        p = self.seg(i) > 0
        if not self.people_on() or not os.path.isfile(self.cpath("people", i, "png")):
            return p
        import cv2
        H, W = p.shape
        full = self.people_ids_full(i, H, W)
        objs = list(self.object_ids())
        obj = np.isin(full, objs) if objs else np.zeros((H, W), bool)
        people = (full > 0) & ~obj                # the numbers that are people, with what they hold
        r = max(2, int(H * 0.01))
        keep = cv2.dilate(people.astype(np.uint8), np.ones((2 * r + 1, 2 * r + 1), np.uint8)) > 0
        if objects:
            out = (p & keep & ~obj) | people
            return out | obj if with_objects else out
        return p & keep & ~obj                    # normals: Sapiens on people only, objects get MoGe

    def people_waiting(self):
        """True when the numbers are found but not checked yet (Generate stops there)."""
        return (self.people_on() and self.s.people_review and self.is_fresh("people")
                and not self.people_edits().get("validated"))

    def needed(self, name):
        """Is this step needed for the passes that are turned on?"""
        s = self.s
        normals, alpha, depth = s.normals_enabled, s.alpha_enabled, s.depth_enabled
        people_alpha = alpha and s.alpha_source in ("matting", "sapiens")
        if name == "people":                      # asked for on the People page, or used by a pass
            return bool(s.people_enabled) and (getattr(self, "want_people", False) or self.needed("seg")
                                               or (alpha and s.alpha_source == "matanyone") or self.only_objects())
        sap = not self.only_objects()             # Sapiens (body seg + crops) only if there are people
        return {
            "plate": True,
            "seg": sap and (normals or people_alpha or self._faces_on()),
            "crops": sap and (normals or (alpha and s.alpha_source == "matting")),   # matting reuses the crop boxes
            "moge": normals or depth,                                     # depth: P needs MoGe's camera
            "merge": normals,
            "stabilize": normals,
            "depth": depth,
            "alpha": alpha,
            "albedo": s.albedo_enabled,
            "write": normals or alpha or depth or s.albedo_enabled,
        }[name]

    def only_objects(self):
        """the segmentation is done and holds only objects (houses, cars...): no Sapiens step at all"""
        s = self.s
        if not s.people_enabled:
            return False
        done = self._read_done("people")
        if not done.get("ids") or done.get("hash") != self._step_hash("people"):
            return False
        e = self.people_edits()
        removed = set(int(x) for x in e.get("removed", []))
        kinds = dict(done.get("kinds", {}), **e.get("kinds", {}))
        kept = [k for k in done["ids"] if k not in removed and str(k) not in e.get("same", {})]
        return bool(kept) and all(kinds.get(str(k)) == "object" for k in kept)

    def _faces_on(self):
        s = self.s
        return not self.only_objects() and s.albedo_enabled and s.albedo_source == "cosmos" and s.cosmos_face_crops

    def active_steps(self):
        return [n for n in STEP_NAMES if self.needed(n)]

    def deps(self, name):
        """(settings, upstream steps) of a step for the current settings."""
        deps, ups = next((d, u) for n, d, u in STEPS if n == name)
        if name == "alpha":                       # LTX does not depend on the crops / seg: keep it cached
            src = self.s.alpha_source
            ups = {"ltx": ("plate",), "matanyone": ("plate", "people"),
                   "matting": ("crops",), "sapiens": ("seg",)}.get(src, ups)
            if src != "matting":
                deps = tuple(d for d in deps if d != "feather_px")
            if src != "ltx":
                deps = tuple(d for d in deps if not d.startswith("ltx_"))
            if src != "matanyone":
                deps = tuple(d for d in deps if not d.startswith("matanyone_"))
        if name == "albedo" and self._faces_on():
            ups = ("plate", "seg")
        if name in ("crops", "merge", "alpha", "albedo") and "seg" in ups and self.s.people_enabled \
                and name != "people" and self.needed("people"):
            ups = tuple(ups) + ("people",)
        return deps, ups

    def _step_hash(self, name):
        deps, ups = self.deps(name)
        vals = {k: getattr(self.s, k, None) for k in deps}
        # the upstream run id (not only its settings): re-running a step makes everything after it stale
        ups_h = {u: self._read_done(u).get("run") for u in ups}
        if name == "people":                      # your click fixes change what SAM3 has to do
            e = self.people_edits()
            vals["keys"] = e.get("keys", [])
            vals["auto"] = e.get("auto", True) is not False
            if not vals["auto"]:                  # clicks only: what SAM 3.1 would look for does not matter
                for k in ("people_prompt", "people_threshold", "people_max", "people_detect_interval"):
                    vals.pop(k, None)
        if "people" in ups:                       # and your keep / same-as decisions what comes after
            e = self.people_edits()
            vals["people_review"] = {"removed": sorted(e.get("removed", [])), "same": e.get("same", {})}
            if self.object_ids():                 # (only when there are some: older caches stay valid)
                vals["people_review"]["objects"] = sorted(self.object_ids())
        if name in ("merge", "alpha") and self.only_objects():
            vals["only_objects"] = sorted(self.object_ids())
        if name == "write" and self.people_on():  # names only go in the EXR header: renaming = rewrite only
            if self.people_edits().get("names") or self.object_ids():
                vals["people_names"] = self.people_edits().get("names", {})
                vals["objects"] = sorted(self.object_ids())
        return _hash({"v": STEP_VERSION.get(name, 1), "deps": vals, "ups": ups_h, "n": self.n, "frames": self.seq.numbers[:1] + self.seq.numbers[-1:]})

    def _read_done(self, name):
        p = os.path.join(self.cache, name, "_done.json")
        if os.path.isfile(p):
            with open(p) as f:
                return json.load(f)
        return {}

    def _mark_done(self, name, h, extra=None):
        d = os.path.join(self.cache, name)
        os.makedirs(d, exist_ok=True)
        info = {"hash": h, "time": time.time(), "run": "%s-%f" % (name, time.time())}
        info.update(extra or {})
        with open(os.path.join(d, "_done.json"), "w") as f:
            json.dump(info, f, indent=1)

    def is_fresh(self, name):
        return self._read_done(name).get("hash") == self._step_hash(name)

    # ------------------------------------------------------------------ run
    def run(self, until=None, force=()):
        """Run every step that is out of date (or in `force`), up to and including `until`."""
        with open(os.path.join(self.out, "laa_job.json"), "w") as f:
            f.write(self.s.to_json())
        if until == "people":                      # asked for on the People page
            self.want_people = True
        t_all = time.time()
        if until is None and not self.needed("write"):
            raise ValueError("Turn on at least one pass.")
        for name in STEP_NAMES:
            self.check()
            if not self.needed(name):
                self.log("[%s] not needed" % name)
                if name == until:
                    break
                continue
            h = self._step_hash(name)
            if name not in force and self._read_done(name).get("hash") == h:
                self.log("[%s] up to date" % name)
            else:
                d = os.path.join(self.cache, name)
                shutil.rmtree(d, ignore_errors=True)
                os.makedirs(d, exist_ok=True)
                t0 = time.time()
                self.log("[%s] running...%s" % (name, (" (GPU: %s)" % vram_text()) if name in GPU_STEPS else ""))
                extra = getattr(self, "step_" + name)() or {}
                self._mark_done(name, self._step_hash(name), extra)
                self.log("[%s] done in %.0fs" % (name, time.time() - t0))
                free_vram()
            if name == "people" and until != "people" and self.people_waiting():
                self.log("[people] check the numbers, then press Continue")
                return "review"
            if name == until:
                break
        self.log("all done in %.0fs -> %s" % (time.time() - t_all, self.out))

    # ------------------------------------------------------------------ steps
    def step_plate(self):
        for i in range(self.n):
            self.check()
            img = seq.read_display_u8(self.seq.path(i), self.s.colorspace)
            seq.write_png(self.cpath("plate", i, "png"), img)
            self.preview("plate", i, img)
            self.progress("plate", i + 1, self.n)
        h, w = img.shape[:2]
        return {"width": w, "height": h}

    def people_frames_dir(self):
        """the plate at the people working size, as JPEGs (what SAM 2.1 reads), kept outside the step
        folder so the People page can click on them before anything is found."""
        import cv2
        from .models.sam3 import work_size
        d = os.path.join(self.cache, "people_frames")
        mark = os.path.join(d, "_plate.json")
        run = self._read_done("plate").get("run")
        try:
            with open(mark) as f:
                if json.load(f).get("run") == run and len(os.listdir(d)) > self.n:
                    return d
        except Exception:
            pass
        shutil.rmtree(d, ignore_errors=True)
        os.makedirs(d, exist_ok=True)
        H, W = self.plate(0).shape[:2]
        w, h = work_size(W, H)
        for i in range(self.n):
            img = cv2.resize(self.plate(i), (w, h), interpolation=cv2.INTER_AREA)
            cv2.imwrite(os.path.join(d, "%05d.jpg" % i), np.ascontiguousarray(img[..., ::-1]),
                        [cv2.IMWRITE_JPEG_QUALITY, 95])
        with open(mark, "w") as f:
            json.dump({"run": run, "size": [w, h]}, f)
        return d

    def key_base(self, key, h, w):
        p = key.get("base")
        if not p:
            return None
        p = p if os.path.isabs(p) else os.path.join(self.out, p)
        if not os.path.isfile(p):
            return None
        b = seq.read_png(p) > 0
        if b.shape != (h, w):
            import cv2
            b = cv2.resize(b.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST) > 0
        return b

    def step_people(self):
        """SAM 3.1 finds everyone (optional), SAM 2.1 follows them through the shot from their best
        frame + every keyframe you clicked (Sammie-Roto style)."""
        import cv2
        from .models.sam3 import work_size
        from .models import sam2 as S2
        from .stages import people as PP
        H, W = self.plate(0).shape[:2]
        w, h = work_size(W, H)
        frames_dir = self.people_frames_dir()
        edits = self.people_edits()
        keys = sorted(edits.get("keys", []), key=lambda k: int(k.get("frame", 0)))
        auto = edits.get("auto", True) is not False
        scale = w / float(W)
        starts = []
        found = {}                                # number -> what it was found as ("person", "car"...)
        if auto:                                  # 1. SAM 3.1: who is there, and where each one is clearest
            from .models.sam3 import SAM3People, prompts
            frames = np.stack([cv2.resize(self.plate(i), (w, h), interpolation=cv2.INTER_AREA) for i in range(self.n)])
            m = SAM3People(self.s, log=self.log, say=self.detail)
            todo = prompts(self.s.people_prompt)
            try:
                off = 0
                for j, prompt in enumerate(todo):       # one SAM 3.1 pass per thing to look for
                    self.detail("Finding: %s (SAM 3.1)" % prompt)
                    A = m.track(frames, None, detect=True, cond=m.encode(prompt) if j else None,
                                progress=lambda a, b, j=j: self.progress(
                                    "people", (j * self.n + min(a, self.n)) // (2 * len(todo)), self.n))
                    got = [(f, k + off, mk) for f, k, mk in PP.anchors(A)]
                    off += A.shape[1]
                    del A
                    starts += got
                    found.update({k: prompt for _, k, _ in got})
                    self.log("[people] SAM 3.1 found %d %s: %s" % (len(got), prompt, ", ".join(
                        "#%d best on frame %s" % (k, self.frame_label(f)) for f, k, _ in got)))
            finally:
                m.release()
                del m
            del frames
        self.check()
        v = S2.SAM2Video(frames_dir, log=self.log)  # 2. SAM 2.1: starts + your keyframes, then propagate
        try:
            for f, k, mk in starts:
                v.add_mask(f, k, mk)
            for key in keys:
                f, k = int(key["frame"]), int(key["target"])
                if 0 <= f < self.n and k > 0:
                    v.set_key(f, k, self.key_base(key, h, w), PP.scaled_points(key, scale))
            self.log("[people] SAM 2.1: %d person(s), %d keyframe(s), following them through the shot"
                     % (len(v.objects()), len(keys)))
            self.detail("Following everyone through the shot (SAM 2.1)")
            off = self.n // 2 if auto else 0
            out = v.propagate(progress=lambda a, b: self.progress("people", off + int(a * (self.n - off) / max(b, 1)),
                                                                  self.n), check=self.check)
        finally:
            v.release()
            del v
        masks = PP.masks_from_propagation(out, self.n, h, w)
        del out
        top = PP.keyed_top(keys)
        ids = PP.to_id_maps(masks, top)
        del masks
        lut = PP.lut(edits)
        count = {}
        for i in range(self.n):
            seq.write_png(self.cpath("people", i, "png"), ids[i])
            for k in np.unique(ids[i]):
                if k:
                    count[int(k)] = count.get(int(k), 0) + 1
            rgb = cv2.resize(self.plate(i), (w, h), interpolation=cv2.INTER_AREA)
            self.preview("people", i, PP.overlay(rgb, lut[ids[i]]))
        self.progress("people", self.n, self.n)
        self.log("[people] %d numbered: %s" % (len(count), ", ".join("%d (%d frames)" % kv for kv in sorted(count.items()))))
        from .models.sam3 import is_human
        found = {k: p for k, p in found.items() if k in count}
        return {"ids": sorted(count), "frames": {str(k): v for k, v in count.items()}, "work": [w, h],
                "found_as": {str(k): p for k, p in found.items()},
                "kinds": {str(k): "object" for k, p in found.items() if not is_human(p)},
                "n_objects": int(ids.max()) if ids.size else 0, "top": [int(k) for k in top]}

    def step_seg(self):
        from .models.sapiens2 import Sapiens2, colorize_seg
        m = Sapiens2("seg", self.s.seg_size, log=self.log)
        m.to_gpu()
        try:
            for i in range(self.n):
                self.check()
                rgb = self.plate(i)
                cid = m.segment(rgb)
                seq.write_png(self.cpath("seg", i, "png"), cid)
                self.preview("seg", i, colorize_seg(rgb, cid))
                self.progress("seg", i + 1, self.n)
        finally:
            m.to_cpu()
            del m

    def _masks(self, i0, i1):
        from .models.sapiens2 import HEAD_CLASSES
        cids = np.stack([self.seg(i) for i in range(i0, i1)])
        return cids > 0, np.isin(cids, HEAD_CLASSES)

    def _all_boxes(self):
        """per-frame crop boxes, fixed over crop_chunk frames (bodies) / tracked (heads)."""
        s = self.s
        boxes, nbody = [], []
        for c0 in range(0, self.n, max(1, s.crop_chunk)):
            c1 = min(self.n, c0 + s.crop_chunk)
            person, head = self._masks(c0, c1)
            ids = None
            if self.people_on():                  # one crop per numbered person, even when they touch
                H, W = person.shape[1:]
                ids = np.stack([self.people_ids_full(i, H, W) for i in range(c0, c1)])
                objs = self.object_ids()
                if objs:                                  # objects: no body / head crop (Sapiens = humans)
                    ids = np.where(np.isin(ids, list(objs)), 0, ids)
                person = person & (ids > 0)               # body pixels (heads); the boxes use the ids
            fb, nb = C.chunk_boxes(person, head, s.crop_padding, s.head_pass, s.head_fraction, ids=ids)
            boxes += fb
            nbody += [nb] * (c1 - c0)
        return boxes, nbody

    def step_crops(self):
        from .models.sapiens2 import Sapiens2
        s = self.s
        boxes, nbody = self._all_boxes()
        with open(os.path.join(self.cache, "crops", "boxes.json"), "w") as f:
            json.dump({"boxes": boxes, "n_body": nbody}, f)
        if not s.normals_enabled:                 # only the boxes (for the matting alpha)
            for i in range(self.n):
                self.preview("boxes", i, D.draw_boxes(self.plate(i), boxes[i], nbody[i], self.person(i)))
                self.progress("crops", i + 1, self.n)
            return
        m = Sapiens2("normal", s.sapiens_size, log=self.log)
        m.to_gpu()
        try:
            for i in range(self.n):
                self.check()
                rgb = self.plate(i)
                H, W = rgb.shape[:2]
                res, tiles = [], []
                for (x0, y0, x1, y1) in boxes[i]:
                    crop = rgb[y0:y1, x0:x1]
                    n = m.normals(crop)
                    res.append(n)
                    tiles.append((crop, n))
                acc = C.paste(res, boxes[i], W, H, s.feather_px, 3, neutral=(0, 0, 1e-4))
                acc = acc / torch.norm(acc, dim=0, keepdim=True).clamp(min=1e-8)
                n_hw = acc.permute(1, 2, 0).numpy()
                np.save(self.cpath("crops", i, "npy"), n_hw.astype(np.float16))
                person = self.person(i)
                self.preview("boxes", i, D.draw_boxes(rgb, boxes[i], nbody[i], person))
                self.preview("tiles", i, D.tile_sheet(tiles))
                self.preview("sapiens", i, D.normals_u8(n_hw))
                self.progress("crops", i + 1, self.n)
        finally:
            m.to_cpu()
            del m
        self.log("[crops] frame 0: %d crops (%d body)" % (len(boxes[0]), nbody[0]))

    def step_moge(self):
        """MoGe-3 in its own process. On some shots the CUDA driver gives up after a dozen frames
        (\"CUDA error: out of memory\" with 20 GB free) and that process loses the GPU for good:
        the frames already written are kept and a fresh process carries on from the next one."""
        from .models import moge as MM
        s = self.s
        level, steps = s.moge_resolution_level, s.moge_refine_steps
        if MM.MoGe.__module__ != MM.__name__ or not torch.cuda.is_available():   # tests / CPU: in this process
            self._moge_frames(0, level, steps)
            return {"resolution_level": level, "refine_steps": steps}
        import subprocess, sys
        from .worker import TAG
        lighter = MM.fallbacks(level, steps)
        start, restarts = 0, 0
        req = os.path.join(self.cache, "moge", "_child.json")
        while start < self.n:
            with open(req, "w") as f:
                json.dump({"settings": dataclasses.asdict(s), "start": start, "level": level, "steps": steps}, f)
            env = dict(os.environ, PYTHONIOENCODING="utf-8")
            flags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
            proc = subprocess.Popen([sys.executable, "-u", "-m", "laa.moge_child", req], cwd=str(config.APP_DIR),
                                    stdout=subprocess.PIPE, stderr=subprocess.STDOUT, env=env, creationflags=flags,
                                    text=True, encoding="utf-8", errors="replace")
            done, last = start, ""
            try:
                for line in proc.stdout:
                    if self._cancel():
                        proc.kill()
                        self.check()
                    line = line.rstrip()
                    if not line.startswith(TAG):
                        if line:
                            self.log(line)
                        continue
                    msg = json.loads(line[len(TAG):])
                    if msg.get("t") == "log":
                        self.log(msg["m"])
                        if "[moge] frame" in msg["m"] or "Error" in msg["m"]:
                            last = msg["m"].strip().splitlines()[-1]
                    elif msg.get("t") == "progress":
                        done = max(done, int(msg["d"]))
                        self._progress("moge", msg["d"], msg["n"])
                    elif msg.get("t") == "detail":
                        self.detail(msg["m"])
            finally:
                rc = proc.wait()
            if rc == 0 and done >= self.n:
                break
            if done > start:                      # got somewhere: same settings, fresh GPU, next frame
                restarts += 1
                self.log("[moge] lost the GPU at frame %d (after %d frames) -> fresh process, carrying on"
                         % (done, done - start))
                start = done
                if restarts > self.n:
                    raise RuntimeError("MoGe keeps losing the GPU (%s)" % last)
                continue
            if lighter:                           # not even one frame: lighter settings, whole shot again
                level, steps = lighter.pop(0)
                self.log("[moge] failed on frame %d (%s) -> resolution %d, refinement %d, whole shot again"
                         % (start, last or "crash", level, steps))
                start = 0
                continue
            raise RuntimeError("MoGe failed on frame %d: %s" % (start, last or "process crashed"))
        self.log("[moge] %s%s" % ("resolution %d, refinement %d" % (level, steps),
                                  ", %d restart(s)" % restarts if restarts else ""))
        return {"resolution_level": level, "refine_steps": steps}

    def _moge_frames(self, start, level, steps):
        """Frames start..n with one MoGe, written to the cache as they come."""
        from .models.moge import MoGe, CV_TO_GL, stats_text
        s = self.s
        m = MoGe(s.moge_model, log=self.log)
        try:
            for i in range(start, self.n):
                self.check()
                try:
                    r = m.infer(self.plate(i), level, s.fov_x, steps)
                except Exception as e:
                    self.log("[moge] frame %d: %s | %s | GPU %s" % (i, str(e).splitlines()[0], stats_text(), vram_text()))
                    raise
                nm = np.where(np.isfinite(r["normal"]), r["normal"], 0.0) * CV_TO_GL
                np.savez(self.cpath("moge", i, "npz"), normal=nm.astype(np.float16), depth=r["depth"].astype(np.float32),
                         mask=r["mask"], intrinsics=r["intrinsics"].astype(np.float32))
                self.preview("moge", i, D.normals_u8(nm))
                self.progress("moge", i + 1, self.n)
        finally:
            m.release()
            del m

    def step_depth(self):
        s = self.s
        from .models.vda import VDA
        m = VDA(s.vda_model, log=self.log)
        frames = np.stack([self.plate(i) for i in range(self.n)])
        self.check()
        d = m.infer(frames, s.vda_input_size, s.vda_max_res, s.vda_fp16,
                    progress=lambda a, b: self.progress("depth", a, b))
        del m, frames
        for i in range(self.n):
            np.save(self.cpath("depth", i, "npy"), d[i].astype(np.float32))
            self.preview("depth", i, D.depth_u8(d[i]))
        return {"source": "vda", "res": list(d.shape[1:])}

    def step_alpha(self):
        s = self.s
        src = s.alpha_source
        if self.only_objects() and src in ("matting", "sapiens"):
            self.log("[alpha] only objects: their SAM 2.1 masks")
            self._alpha_objects_only()
        elif src == "ltx":
            self._alpha_ltx()
        elif src == "matanyone":
            return {"source": src, "numbers": self._alpha_matanyone()}
        elif src == "matting":
            from .models.sapiens2 import Sapiens2
            with open(os.path.join(self.cache, "crops", "boxes.json")) as f:
                bx = json.load(f)
            m = Sapiens2("matting", "1b", log=self.log)
            m.to_gpu()
            try:
                for i in range(self.n):
                    self.check()
                    rgb = self.plate(i)
                    H, W = rgb.shape[:2]
                    res, tiles = [], []
                    for (x0, y0, x1, y1) in bx["boxes"][i]:
                        crop = rgb[y0:y1, x0:x1]
                        a = torch.from_numpy(m.matte(crop))[None]
                        res.append(a)
                        tiles.append((crop, a))
                    a = C.paste(res, bx["boxes"][i], W, H, s.feather_px, 1)[0].numpy() if res else np.zeros((H, W), np.float32)
                    a = np.maximum(a, self.object_alpha(i, H, W))   # Sapiens matting only knows people
                    self._save_alpha(i, a)
                    self.preview("alpha_tiles", i, D.tile_sheet(tiles))
                    self.progress("alpha", i + 1, self.n)
            finally:
                m.to_cpu()
                del m
        else:                                     # Sapiens2 segmentation, hard edge
            for i in range(self.n):
                self._save_alpha(i, self.person(i, objects=True).astype(np.float32))
                self.progress("alpha", i + 1, self.n)
        return {"source": src}

    def _alpha_ltx(self):
        """Foreground: LTX Alpha Gen on the whole frame. It decides by itself what the foreground is
        (no prompt, no mask: that is how the model works), so the selection is not used here."""
        import cv2
        from .models.ltx_alpha import LTXAlpha, chunks_with_overlap
        s = self.s
        H, W = self.plate(0).shape[:2]
        canvas = np.zeros((self.n, H, W), np.float16)
        jobs = chunks_with_overlap(self.n, max(1, s.ltx_chunk), max(0, s.ltx_overlap))
        self.progress("alpha", 0, self.n)
        m = None                                      # LTX (~38 GB) is only loaded if a chunk is missing
        try:
            for ji, (lo, hi, k0, k1) in enumerate(jobs):
                self.check()
                label = "(frames %s-%s)" % (self.frame_label(k0), self.frame_label(k1 - 1))
                # finished LTX chunks are kept outside the step folder: a crash or a Cancel later in
                # the step does not throw away 10 minutes of diffusion
                key = _hash({"ltx": {k: getattr(s, k) for k in ("ltx_unet", "ltx_lora", "ltx_seed", "ltx_fps",
                                                                "ltx_max_width", "ltx_max_height")},
                             "full": True, "lo": lo, "hi": hi, "plate": self._read_done("plate").get("run")})
                kept = os.path.join(self.cache, "alpha_ltx_chunks", key + ".npy")
                if os.path.isfile(kept):
                    self.log("[alpha] LTX %s: reused (already computed)" % label)
                    a = np.load(kept).astype(np.float32)
                else:
                    self.log("[alpha] LTX %s, whole frame" % label)
                    if m is None:
                        m = LTXAlpha(s, log=self.log, say=self.detail)
                    a = m.matte(np.stack([self.plate(i) for i in range(lo, hi)]), label=label,
                                progress=lambda f: self.progress("alpha", int(self.n * (ji + f) / len(jobs)), self.n))
                    os.makedirs(os.path.dirname(kept), exist_ok=True)
                    np.save(kept, a.astype(np.float16))
                for i in range(k0, k1):
                    ai = a[i - lo]
                    if ai.shape != (H, W):
                        ai = cv2.resize(ai, (W, H), interpolation=cv2.INTER_LINEAR)
                    canvas[i] = ai.astype(np.float16)
                self.progress("alpha", int(self.n * (ji + 1) / len(jobs)), self.n)
        finally:
            if m is not None:
                m.release()
                del m
        for i in range(self.n):
            self._save_alpha(i, canvas[i].astype(np.float32))

    def alpha_numbers(self):
        """the numbers that get their own alpha (Selection mode): kept people and objects"""
        if not self.people_on():
            return []
        done = self._read_done("people")
        e = self.people_edits()
        removed = set(int(x) for x in e.get("removed", []))
        same = {int(a): int(b) for a, b in e.get("same", {}).items()}
        return [k for k in done.get("ids", []) if k not in removed and k not in same]

    def _alpha_matanyone(self):
        """Selection: MatAnyone 2 on each number, guided by its SAM 2.1 mask, so the alpha is exactly
        what you selected (a person with their rifle, a car...), nothing else."""
        import cv2
        from .models.matanyone2 import MatAnyone2
        from .stages import people as PP
        s = self.s
        H, W = self.plate(0).shape[:2]
        nums = self.alpha_numbers()
        if not nums:
            raise RuntimeError("Alpha 'Selection' needs something selected on page 1 (Segmentation), "
                               "or choose Alpha > Foreground (LTX).")
        sc = min(1.0, float(s.matanyone_max_side) / min(H, W)) if s.matanyone_max_side > 0 else 1.0
        h, w = max(16, int(round(H * sc))) // 2 * 2, max(16, int(round(W * sc))) // 2 * 2
        lut = PP.lut(self.people_edits())             # "same as": both numbers become one
        ids = np.stack([cv2.resize(lut[self.people_ids(i, final=False)], (w, h), interpolation=cv2.INTER_NEAREST)
                        for i in range(self.n)])
        frames = None
        m = None
        out_dir = os.path.join(self.cache, "alpha", "numbers")
        os.makedirs(out_dir, exist_ok=True)
        combined = np.zeros((self.n, h, w), np.float32)
        try:
            for j, k in enumerate(nums):
                self.check()
                masks = ids == k
                areas = masks.reshape(self.n, -1).sum(1)
                if not areas.any():
                    continue
                anchor = int(np.argmax(areas))
                key = _hash({"ma2": [s.matanyone_max_side, s.matanyone_warmup, s.matanyone_grow], "k": k,
                             "people": self._read_done("people").get("run"), "same": self.people_edits().get("same", {}),
                             "plate": self._read_done("plate").get("run"), "v": 1})
                kept = os.path.join(self.cache, "alpha_ma2", key + ".npy")
                name = "%d of %d" % (j + 1, len(nums))
                if os.path.isfile(kept):
                    self.log("[alpha] MatAnyone 2, number %d: reused (already computed)" % k)
                    a8 = np.load(kept)
                else:
                    if m is None:
                        self.detail("Loading MatAnyone 2")
                        m = MatAnyone2(log=self.log)
                    if frames is None:
                        frames = np.stack([cv2.resize(self.plate(i), (w, h), interpolation=cv2.INTER_AREA)
                                           for i in range(self.n)])
                    self.log("[alpha] MatAnyone 2, number %d (%s), from frame %s at %dx%d"
                             % (k, name, self.frame_label(anchor), w, h))
                    self.detail("MatAnyone 2: %s" % name)
                    a = m.matte(frames, masks, anchor, warmup=s.matanyone_warmup, grow=s.matanyone_grow,
                                check=self.check, progress=lambda d, n_, j=j: self.progress(
                                    "alpha", int(self.n * (j + d / max(n_, 1)) / len(nums)), self.n))
                    a8 = np.clip(a * 255.0 + 0.5, 0, 255).astype(np.uint8)
                    os.makedirs(os.path.dirname(kept), exist_ok=True)
                    np.save(kept, a8)
                combined = np.maximum(combined, a8.astype(np.float32) / 255.0)
                np.save(os.path.join(out_dir, "%d.npy" % k), a8)
        finally:
            if m is not None:
                m.release()
                del m
        for i in range(self.n):
            a = combined[i]
            if (h, w) != (H, W):
                a = cv2.resize(a, (W, H), interpolation=cv2.INTER_LINEAR)
            self._save_alpha(i, a)
        self.progress("alpha", self.n, self.n)
        return nums

    def number_alpha(self, i, k, H, W):
        """alpha of number k alone on frame i (Selection mode), for the EXR layer alpha_<k>"""
        import cv2
        p = os.path.join(self.cache, "alpha", "numbers", "%d.npy" % k)
        if not os.path.isfile(p):
            return None
        a = np.load(p, mmap_mode="r")[i].astype(np.float32) / 255.0
        return cv2.resize(a, (W, H), interpolation=cv2.INTER_LINEAR) if a.shape != (H, W) else a

    def object_alpha(self, i, H, W):
        """objects (car, house, rifle...): their SAM 2.1 mask, edge softened by a pixel or two"""
        import cv2
        objs = self.object_ids()
        if not objs:
            return np.zeros((H, W), np.float32)
        m = np.isin(self.people_ids_full(i, H, W), list(objs)).astype(np.float32)
        return cv2.GaussianBlur(m, (0, 0), max(0.8, H / 1200.0))

    def _alpha_objects_only(self):
        for i in range(self.n):
            self.check()
            H, W = self.plate(i).shape[:2]
            self._save_alpha(i, self.object_alpha(i, H, W))
            self.progress("alpha", i + 1, self.n)

    def _save_alpha(self, i, a):
        a16 = np.clip(a * 65535.0 + 0.5, 0, 65535).astype(np.uint16)
        seq.write_png(self.cpath("alpha", i, "png"), a16)
        self.preview("alpha", i, (np.repeat(a[..., None], 3, -1) * 255 + 0.5).astype(np.uint8))

    def alpha(self, i):
        return seq.read_png(self.cpath("alpha", i, "png")).astype(np.float32) / 65535.0

    def step_albedo(self):
        if self.s.albedo_source == "cosmos":
            return self._albedo_cosmos()
        return self._albedo_marigold()

    def _albedo_cosmos(self):
        from .models.cosmos import CosmosRenderer, letterbox, unletterbox, decode
        from .io.sequence import linear_to_srgb
        d = os.path.join(self.cache, "albedo")
        din, dout = os.path.join(d, "cosmos_in"), os.path.join(d, "cosmos_out")
        box = None
        for i in range(self.n):                       # 1. plate -> 1280x704, no stretch
            img, box = letterbox(self.plate(i))
            seq.write_png(os.path.join(din, "%05d.png" % i), img)
        if self.s.cosmos_backend == "wsl":
            m = CosmosRenderer(self.s, log=self.log, say=self.detail)
        else:
            from .models.cosmos_native import CosmosNative
            m = CosmosNative(self.s, log=self.log, say=self.detail)
        passes = m.passes()
        faces = self.s.cosmos_face_crops
        tracks = self._face_tracks() if faces else []
        runs = 1 + len(tracks)                        # progress over the full frame + every face
        self.progress("albedo", 0, self.n)
        m.run(din, dout, cancel=self._cancel,       # 2. Cosmos on the full frame (57-frame chunks)
              progress=lambda dn, tot: self.progress("albedo", int(self.n * dn / max(tot, 1) / runs), self.n))
        layers = []
        for i in range(self.n):                       # 3. back to plate size, float layers
            self.check()
            H, W = self.plate(i).shape[:2]
            out = {}
            for p in passes:
                u8 = seq.read_png(os.path.join(dout, p, "%05d.png" % i))
                if u8.ndim == 2:
                    u8 = np.repeat(u8[..., None], 3, -1)
                out.update(decode(p, unletterbox(u8, box, H, W)))
            np.savez(self.cpath("albedo", i, "npz"), **{k: v.astype(np.float16) for k, v in out.items()})
            prev = (np.clip(linear_to_srgb(out["albedo"]), 0, 1) * 255 + 0.5).astype(np.uint8)
            self.preview("albedo", i, prev)
            if faces:
                self.preview("albedo_full", i, prev)
            layers = list(out)
        shutil.rmtree(din, ignore_errors=True)
        if tracks:                                    # 4. faces: Cosmos again on a crop that follows each head
            self._albedo_faces(m, tracks, passes, runs)
        return {"source": "cosmos", "layers": layers, "faces": len(tracks)}

    def _face_tracks(self):
        from .models.sapiens2 import HEAD_CLASSES
        if self.only_objects():                   # no people, no faces
            return []
        heads = np.stack([np.isin(self.seg(i), HEAD_CLASSES) for i in range(self.n)])
        H, W = heads.shape[1:]
        if self.people_on():                      # one head per numbered person (two heads can touch)
            ids = np.stack([self.people_ids(i) for i in range(self.n)])
            tracks = []
            for k in [int(v) for v in np.unique(ids) if v]:
                import cv2
                mk = np.stack([cv2.resize((ids[i] == k).astype(np.uint8), (W, H), interpolation=cv2.INTER_NEAREST) > 0
                               for i in range(self.n)])
                tk = C.head_tracks(heads & mk, W, H, aspect=1280.0 / 704.0, head_share=self.s.cosmos_face_size)
                if tk:
                    best = max(tk, key=lambda t: sum(t["present"]))
                    best["id"] = k
                    tracks.append(best)
                del mk
            del ids
        else:
            tracks = C.head_tracks(heads, W, H, aspect=1280.0 / 704.0, head_share=self.s.cosmos_face_size)
        self.log("[albedo] %d face(s) to refine" % len(tracks))
        with open(os.path.join(self.cache, "albedo", "faces.json"), "w") as f:
            json.dump(tracks, f)
        return tracks

    def _albedo_faces(self, m, tracks, passes, runs):
        """Cosmos on each head crop (~3-4x more pixels on the face), pasted back on the person only,
        colour-matched to the full-frame albedo so no box shows."""
        import cv2
        from .models.cosmos import decode, W_IN, H_IN
        from .models.sapiens2 import HEAD_CLASSES
        from .io.sequence import linear_to_srgb
        d = os.path.join(self.cache, "albedo")
        H, W = self.plate(0).shape[:2]
        pasted = {}                                   # frame -> list of (box, layers, weight)
        for ti, t in enumerate(tracks):
            self.check()
            din, dout = os.path.join(d, "face%d_in" % ti), os.path.join(d, "face%d_out" % ti)
            for i, (x0, y0, x1, y1) in enumerate(t["boxes"]):
                crop = self.plate(i)[y0:y1, x0:x1]
                seq.write_png(os.path.join(din, "%05d.png" % i),
                              cv2.resize(crop, (W_IN, H_IN), interpolation=cv2.INTER_CUBIC))
            self.detail("Face %d of %d" % (ti + 1, len(tracks)))
            k0 = self.n * (1 + ti) / runs
            m.run(din, dout, cancel=self._cancel,
                  progress=lambda dn, tot: self.progress("albedo", int(k0 + self.n * dn / max(tot, 1) / runs), self.n))
            # read back + where to paste: the person inside the box, softened, away from the box edge
            seg_m = []
            for i, (x0, y0, x1, y1) in enumerate(t["boxes"]):
                bw, bh = x1 - x0, y1 - y0
                lay = {}
                for p in passes:
                    if p in ("normal", "depth"):
                        continue
                    u8 = seq.read_png(os.path.join(dout, p, "%05d.png" % i))
                    if u8.ndim == 2:
                        u8 = np.repeat(u8[..., None], 3, -1)
                    lay.update(decode(p, cv2.resize(u8, (bw, bh), interpolation=cv2.INTER_AREA)))
                # only the head (face/neck, hair, lips...): the body stays from the full-frame pass,
                # a close-up crop full of wall can misread a dark jacket as grey
                # the blend happens INSIDE the head silhouette (eroded, then softened, then cut by the
                # silhouette): nothing of the crop's background or jacket can leak around the head
                head = np.isin(self.seg(i)[y0:y1, x0:x1], HEAD_CLASSES)
                if "id" in t:                         # only this person's head, not a neighbour's
                    head &= self.people_ids_full(i, H, W)[y0:y1, x0:x1] == t["id"]
                head = head.astype(np.uint8)
                r = max(1, int(bh * self.s.cosmos_face_edge))
                inner = cv2.erode(head, np.ones((2 * r + 1, 2 * r + 1), np.uint8)).astype(np.float32)
                inner = cv2.GaussianBlur(inner, (0, 0), r * 0.7)
                w = np.clip(inner, 0, 1) * head * C.feather([x0, y0, x1, y1], W, H, max(4, int(bh * 0.08)))
                seg_m.append(w)
                pasted.setdefault(i, []).append(((x0, y0, x1, y1), lay, w))
            # one colour gain per face for the whole shot: crop albedo -> full-frame albedo level
            num, den = np.zeros(3), np.zeros(3)
            for i, (x0, y0, x1, y1) in enumerate(t["boxes"]):
                full = np.load(self.cpath("albedo", i, "npz"))["albedo"][y0:y1, x0:x1].astype(np.float32)
                sel = seg_m[i] > 0.5
                if sel.sum() > 50:
                    num += full[sel].mean(0)
                    den += pasted[i][-1][1]["albedo"][sel].mean(0)
            gain = np.clip(np.where(den > 1e-6, num / np.maximum(den, 1e-6), 1.0), 0.5, 2.0).astype(np.float32)
            self.log("[albedo] face %d colour gain %.2f %.2f %.2f" % ((ti + 1,) + tuple(gain)))
            for i in range(self.n):
                pasted[i][-1][1]["albedo"] = pasted[i][-1][1]["albedo"] * gain
            shutil.rmtree(din, ignore_errors=True)
        # paste every face into the full-frame layers
        for i in range(self.n):
            npz = dict(np.load(self.cpath("albedo", i, "npz")))
            out = {k: v.astype(np.float32) for k, v in npz.items()}
            tiles = []
            for (x0, y0, x1, y1), lay, w in pasted.get(i, []):
                for k, v in lay.items():
                    if k not in out:
                        continue
                    ww = w[..., None] if v.ndim == 3 else w
                    out[k][y0:y1, x0:x1] = out[k][y0:y1, x0:x1] * (1 - ww) + v * ww
                tiles.append((self.plate(i)[y0:y1, x0:x1],
                              torch.from_numpy(np.clip(linear_to_srgb(lay["albedo"]), 0, 1)).permute(2, 0, 1) * 2 - 1))
            np.savez(self.cpath("albedo", i, "npz"), **{k: v.astype(np.float16) for k, v in out.items()})
            self.preview("albedo", i, (np.clip(linear_to_srgb(out["albedo"]), 0, 1) * 255 + 0.5).astype(np.uint8))
            if tiles:
                self.preview("albedo_faces", i, D.tile_sheet(tiles, th=360, aspect="keep"))
        self.progress("albedo", self.n, self.n)

    def _albedo_marigold(self):
        s = self.s
        from .models.albedo import MarigoldIID
        m = MarigoldIID(s.albedo_variant, s.albedo_steps, log=self.log)
        m.to_gpu()
        try:
            for i in range(self.n):
                self.check()
                r = m.infer(self.plate(i), seed=0)
                np.savez(self.cpath("albedo", i, "npz"), **{k: v.astype(np.float16) for k, v in r.items()})
                if "albedo" in r:
                    from .io.sequence import linear_to_srgb
                    self.preview("albedo", i, (np.clip(linear_to_srgb(r["albedo"]), 0, 1) * 255 + 0.5).astype(np.uint8))
                self.progress("albedo", i + 1, self.n)
        finally:
            m.to_cpu()
            del m
        return {"source": "marigold", "layers": list(r.keys())}

    def step_merge(self):
        s = self.s
        signs = None
        if self.only_objects():                   # nobody: the normals are MoGe's everywhere
            self.log("[merge] only objects selected: MoGe normals everywhere")
            for i in range(self.n):
                nm = np.load(self.cpath("moge", i, "npz"))["normal"].astype(np.float32)
                np.save(self.cpath("merge", i, "npy"), nm.astype(np.float16))
                np.save(self.cpath("merge", i, "mask.npy"), np.zeros(nm.shape[:2], np.float16))
                self.preview("merged", i, D.normals_u8(nm))
                self.progress("merge", i + 1, self.n)
            return {"axis_signs": [1.0, 1.0, 1.0]}
        # axis signs once for the sequence: the frame with the most person pixels among the first ones
        probe = max(range(min(self.n, 25)), key=lambda i: self.person(i).sum())
        sel = C.soft_mask(self.person(probe)[None], s.mask_erode_px, s.mask_blur_px)[0] > 0.5
        nd = np.load(self.cpath("crops", probe, "npy")).astype(np.float32)
        nm = np.load(self.cpath("moge", probe, "npz"))["normal"].astype(np.float32)
        if sel.sum() < 100:
            sel = np.ones(sel.shape, bool)
        corr = (nd[sel] * nm[sel]).sum(axis=0)
        signs = np.where(corr < 0, -1.0, 1.0).astype(np.float32)
        self.log("[merge] Sapiens axis signs vs MoGe: %+d %+d %+d" % tuple(int(v) for v in signs))
        for i in range(self.n):
            self.check()
            m = C.soft_mask(self.person(i)[None], s.mask_erode_px, s.mask_blur_px)[0]
            nd = np.load(self.cpath("crops", i, "npy")).astype(np.float32) * signs
            nm = np.load(self.cpath("moge", i, "npz"))["normal"].astype(np.float32)
            mix = nd * m[..., None] + nm * (1.0 - m[..., None])
            mix = mix / np.maximum(np.linalg.norm(mix, axis=-1, keepdims=True), 1e-6)
            np.save(self.cpath("merge", i, "npy"), mix.astype(np.float16))
            np.save(self.cpath("merge", i, "mask.npy"), m.astype(np.float16))
            self.preview("blend", i, D.gray_u8(m, 0, 1))
            self.preview("merged", i, D.normals_u8(mix))
            heat, mean, p95 = D.angle_heatmap(mix, nd, m > 0.99)
            self.preview("checkA", i, heat)
            if i == 0:
                self.log("[merge] CHECK A frame %s: mean %.2f deg, 95%% < %.2f deg (should be ~0)"
                         % (self.frame_label(i), mean, p95))
            self.progress("merge", i + 1, self.n)
        return {"axis_signs": signs.tolist()}

    def step_stabilize(self):
        s = self.s
        get_v = lambda i: np.load(self.cpath("merge", i, "npy")).astype(np.float32)
        get_p = lambda i: self.plate(i).astype(np.float32) / 255.0

        def put(i, r):
            np.save(self.cpath("stabilize", i, "npy"), r.astype(np.float16))
            self.preview("final", i, D.normals_u8(r))
            heat, _, _ = D.angle_heatmap(r, get_v(i), None, max_deg=10.0)
            self.preview("checkB", i, heat)

        temporal.stabilize_stream(self.n, get_v, get_p, put, s.stab_radius, s.stab_sigma, s.stab_max_angle,
                                  normals=True, progress=lambda a, b: self.progress("stabilize", a, b),
                                  cancel=self._cancel, log=self.log)

    def exr_path(self, i):
        return os.path.join(self.out, "%s_aovs.%s.exr" % (self.name, self.frame_label(i)))

    def step_write(self):
        s = self.s
        npt = HALF if s.half_float else FLOAT
        normals, depth, alpha, albedo = s.normals_enabled, s.depth_enabled, s.alpha_enabled, s.albedo_enabled
        moge_on = self.needed("moge")
        signs = self._read_done("merge").get("axis_signs") if normals else None
        uu = vv = None
        for i in range(self.n):
            self.check()
            H, W = self.plate(i).shape[:2]
            ch = {}
            attrs = {"laa/source": "LiveActionAOVs", "laa/frame": float(self.seq.numbers[i])}
            if self.people_on():                  # people.id value -> the name you gave them
                e = self.people_edits()
                names = {k: v for k, v in e.get("names", {}).items()
                         if v and int(k) not in set(int(x) for x in e.get("removed", [])) and k not in e.get("same", {})}
                if names:
                    attrs["people/names"] = json.dumps(names, sort_keys=True)
                objs = sorted(self.object_ids())
                if objs:
                    attrs["people/objects"] = ",".join(str(k) for k in objs)

            def put(layer, a, pt=npt):
                if a.ndim == 2:
                    ch["%s.Y" % layer] = (a.astype(np.float32), pt)
                    return
                for k, c in enumerate("RGB"):
                    ch["%s.%s" % (layer, c)] = (a[..., k].astype(np.float32), pt)

            g = np.load(self.cpath("moge", i, "npz")) if moge_on else None
            if normals:
                put("N", np.load(self.cpath("stabilize", i, "npy")).astype(np.float32))
                if s.comparison_layers:
                    put("Nraw", np.load(self.cpath("merge", i, "npy")).astype(np.float32))
                    put("Nmoge", g["normal"].astype(np.float32))
            if depth:
                z_m = g["depth"].astype(np.float32)
                valid = np.isfinite(z_m) & (z_m > 0)
                z_m = np.where(valid, z_m, 0.0).astype(np.float32)
                d = _resize(np.load(self.cpath("depth", i, "npy")), H, W)
                dvalid = np.isfinite(d) & (d > 0)
                d = np.where(dvalid, d, 0.0).astype(np.float32)
                ch["depth.Z"] = (d, FLOAT)
                if s.comparison_layers:
                    ch["depth_moge.Z"] = (z_m, FLOAT)
                    ch["moge_valid.A"] = ((g["mask"].astype(np.float32) * valid * dvalid), npt)
                both = (d > 0) & (z_m > 0)
                if both.any():
                    attrs["depth/vda_over_moge"] = float(np.median(d[both] / z_m[both]))
                # P: pinhole unprojection with MoGe's (fixed-FOV) intrinsics, OpenCV -> Nuke
                K = g["intrinsics"]
                fx, fy, cx, cy = float(K[0, 0]) * W, float(K[1, 1]) * H, float(K[0, 2]) * W, float(K[1, 2]) * H
                if uu is None:
                    uu, vv = np.meshgrid(np.arange(W, dtype=np.float32) + 0.5, np.arange(H, dtype=np.float32) + 0.5)
                P = np.stack(((uu - cx) / fx * d, (vv - cy) / fy * d, d), -1) * np.array([1, -1, -1], np.float32)
                put("P", P, FLOAT)
            if g is not None:
                K = g["intrinsics"]
                attrs["moge/fov_x"] = math.degrees(2 * math.atan(0.5 / float(K[0, 0])))
                attrs["moge/fov_y"] = math.degrees(2 * math.atan(0.5 / float(K[1, 1])))
                attrs["moge/focal_px"] = float(K[0, 0]) * W
            if alpha:
                ch["A"] = (_resize(self.alpha(i), H, W).clip(0, 1).astype(np.float32), npt)
                for k in self._read_done("alpha").get("numbers", []):     # Selection: one alpha per number
                    ak = self.number_alpha(i, k, H, W)
                    if ak is not None:
                        ch["alpha_%d.A" % k] = (ak.clip(0, 1).astype(np.float32), npt)
            if self.people_on() and os.path.isfile(self.cpath("people", i, "png")):
                # person number per pixel (0 nobody): Keyer / Expression on people.id picks one person
                ch["people.id"] = (self.people_ids_full(i, H, W).astype(np.float32), FLOAT)
            if albedo:
                al = np.load(self.cpath("albedo", i, "npz"))
                for k in al.files:
                    put(k, _resize(al[k].astype(np.float32), H, W))
            if signs:
                attrs["detail/axis_signs"] = "%+d %+d %+d" % tuple(int(v) for v in signs)
            if normals:
                attrs["detail/stabilize_radius"] = float(s.stab_radius)
            write_exr(self.exr_path(i), ch, attrs)
            self.progress("write", i + 1, self.n)
        self.log("[write] %d EXR: %s" % (self.n, self.exr_path(0).replace(self.frame_label(0), "####")))


def _resize(a, H, W):
    if a.shape[:2] == (H, W):
        return a.astype(np.float32)
    import cv2
    return cv2.resize(a.astype(np.float32), (W, H), interpolation=cv2.INTER_LINEAR)

