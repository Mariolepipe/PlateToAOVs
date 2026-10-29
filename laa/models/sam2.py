"""
SAM 2.1 (Meta) video predictor, used the way Sammie-Roto uses it: points on any frame, per person
("keyframes"), an instant mask on that frame, then one propagation through the shot that uses ALL
the frames you clicked as references at the same time.

Code: the official repo, cloned into vendor/sam2 (download_sam2.bat). Weights:
models/sam2/sam2.1_hiera_large.pt (from Meta, no access request).
"""

import contextlib
import os
import sys

import numpy as np
import torch

from .. import config

CKPT = "sam2.1_hiera_large.pt"
CKPT_URL = "https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_large.pt"
CFG = "configs/sam2.1/sam2.1_hiera_l.yaml"


def repo_dir():
    return os.path.join(str(config.APP_DIR), "vendor", "sam2")


def _import():
    d = repo_dir()
    if not os.path.isfile(os.path.join(d, "sam2", "build_sam.py")):
        raise RuntimeError("SAM 2 code missing in vendor/sam2: run download_sam2.bat.")
    if d not in sys.path:
        sys.path.insert(0, d)
    from sam2.build_sam import build_sam2_video_predictor
    return build_sam2_video_predictor


class SAM2Video(object):
    """One shot: frames_dir holds 00000.jpg, 00001.jpg... (working size)."""

    def __init__(self, frames_dir, log=print, ckpt=None, cfg=CFG, device=None):
        build = _import()
        if ckpt is None:
            ckpt = config.require_model(CKPT, "sam2", "Download it in the app: Models, bottom right.")
        elif ckpt is False:                         # tests: the real code, random weights
            ckpt = None
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        log("loading SAM2 %s" % (os.path.basename(ckpt) if ckpt else "(no weights)"))
        self.pred = build(cfg, ckpt, device=self.device)
        n = len([f for f in os.listdir(frames_dir) if f.lower().endswith(".jpg")])
        with self._ctx():
            self.state = self.pred.init_state(frames_dir, offload_video_to_cpu=True,
                                              offload_state_to_cpu=n > 150)
        self.n = n

    def _ctx(self):
        stack = contextlib.ExitStack()
        stack.enter_context(torch.inference_mode())
        # SAM2 keeps its memory in bfloat16: it always runs under autocast (as in Meta's examples)
        stack.enter_context(torch.autocast(self.device.type, dtype=torch.bfloat16))
        return stack

    def _mask(self, obj_ids, logits, obj):
        obj_ids = list(obj_ids)
        if obj not in obj_ids:
            return None
        return (logits[obj_ids.index(obj)] > 0.0).float().cpu().numpy().squeeze().astype(bool)

    def add_mask(self, frame, obj, mask):
        """a starting mask for a person on a frame (SAM3's find-everyone result)"""
        with self._ctx():
            _, ids, logits = self.pred.add_new_mask(self.state, int(frame), int(obj), np.asarray(mask, bool))
        return self._mask(ids, logits, int(obj))

    def set_key(self, frame, obj, base=None, points=()):
        """
        rebuild what you did on one frame for one person, from scratch: their mask on that frame when
        you started clicking (base, or None for a new person), then your clicks replayed one by one
        (each click refines the previous result, exactly like when you clicked).
        points: [(x, y, 1 = this person | 0 = not this person)] in frame pixels
        -> (h, w) bool mask on that frame
        """
        frame, obj = int(frame), int(obj)
        st = self.state
        with self._ctx():
            if obj in st["obj_id_to_idx"]:
                self.pred.clear_all_prompts_in_frame(st, frame, obj, need_output=False)
                idx = st["obj_id_to_idx"][obj]
                st["output_dict_per_obj"][idx]["non_cond_frame_outputs"].pop(frame, None)
                st["temp_output_dict_per_obj"][idx]["non_cond_frame_outputs"].pop(frame, None)
            m = None
            if base is not None and np.asarray(base).any():
                # a person found by SAM 3.1 (or propagated): each click adds / removes what is under
                # it, the rest of their mask stays as it is. (Clicks on top of the mask itself make
                # SAM 2.1 use it as a mask prompt, which comes out as a grid of dots.)
                m = self._edit(frame, obj, np.asarray(base, bool), points)
                _, ids, logits = self.pred.add_new_mask(st, frame, obj, m)
                return self._mask(ids, logits, obj)
            pts = [(float(x), float(y)) for x, y, _ in points]
            lab = [int(l) for _, _, l in points]
            for i in range(len(pts)):
                _, ids, logits = self.pred.add_new_points_or_box(
                    st, frame, obj, points=np.array(pts[:i + 1], np.float32),
                    labels=np.array(lab[:i + 1], np.int32), clear_old_points=True)
                m = self._mask(ids, logits, obj)
        return m

    def _edit(self, frame, obj, base, points):
        """base mask + clicks -> mask: left click = add the part under it, right click = remove it"""
        st, tmp = self.state, 100000 + obj               # a scratch object, removed right after
        m = base.copy()
        h, w = m.shape
        try:
            for x, y, l in points:
                _, ids, logits = self.pred.add_new_points_or_box(
                    st, frame, tmp, points=np.array([[float(x), float(y)]], np.float32),
                    labels=np.array([1], np.int32), clear_old_points=True)
                r = self._mask(ids, logits, tmp)
                if r is None or r.shape != m.shape:
                    continue
                if l:
                    m |= r
                    continue
                cut = r & m                          # not this person: what SAM sees there, out of them
                if cut.sum() > 0.5 * max(m.sum(), 1):    # that is most of them (they were clicked on):
                    import cv2                           # only the piece under the click
                    _, lab = cv2.connectedComponents(cut.astype(np.uint8))
                    xi, yi = min(w - 1, max(0, int(x))), min(h - 1, max(0, int(y)))
                    cut = lab == lab[yi, xi] if lab[yi, xi] else np.zeros_like(cut)
                    if cut.sum() > 0.5 * max(m.sum(), 1):
                        continue
                m &= ~cut
        finally:
            self.pred.remove_object(st, tmp, need_output=False)
        return m

    def objects(self):
        return list(self.state["obj_ids"])

    def propagate(self, progress=None, check=lambda: None):
        """-> {frame: {obj: (h, w) bool}} for every frame. Two passes: forward from the first keyframe
        to the end, backward from the last keyframe to frame 0. Each frame of each person keeps the
        pass that comes from their nearest keyframe, so a click on the last frame also fixes the
        frames before it (not only the ones after)."""
        if not self.state["obj_ids"]:
            return {}
        st = self.state
        self.pred.propagate_in_video_preflight(st)      # your clicks become keyframes
        conds = {self.state["obj_idx_to_id"][i]: sorted(d["cond_frame_outputs"])
                 for i, d in st["output_dict_per_obj"].items()}
        frames = [t for c in conds.values() for t in c]
        first, last = min(frames), max(frames)
        total = (self.n - first) + (last + 1)
        done = [0]

        def run(start, reverse):
            res = {}
            for f, ids, logits in self.pred.propagate_in_video(st, start_frame_idx=start, reverse=reverse):
                check()
                masks = (logits > 0.0).float().cpu().numpy()[:, 0].astype(bool)
                res[int(f)] = {int(o): masks[k] for k, o in enumerate(ids)}
                done[0] += 1
                if progress:
                    progress(min(done[0], total), total)
            return res

        with self._ctx():
            fwd = run(first, False)
            for d in st["output_dict_per_obj"].values():   # the backward pass starts clean
                d["non_cond_frame_outputs"].clear()
            bwd = run(last, True) if last > 0 else {}

        def pick(f, o):
            c = conds.get(o, [])
            d_f = min((f - t for t in c if t <= f), default=None)
            d_b = min((t - f for t in c if t >= f), default=None)
            if f in fwd and o in fwd[f] and (d_b is None or f not in bwd or (d_f is not None and d_f <= d_b)):
                return fwd[f][o]
            return bwd.get(f, {}).get(o)

        out = {}
        for f in range(self.n):
            objs = set(fwd.get(f, {})) | set(bwd.get(f, {}))
            out[f] = {o: m for o in objs for m in [pick(f, o)] if m is not None}
        return out

    def release(self):
        try:
            self.pred.reset_state(self.state)
        except Exception:
            pass
        self.pred = self.state = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def download(log=print):
    """vendor/sam2 (official code) + the large checkpoint."""
    import subprocess
    import urllib.request
    d = repo_dir()
    if not os.path.isfile(os.path.join(d, "sam2", "build_sam.py")):
        log("cloning facebookresearch/sam2 into %s" % d)
        subprocess.check_call(["git", "clone", "--depth", "1", "https://github.com/facebookresearch/sam2.git", d])
    dst_dir = config.default_download_dir("sam2")
    dst = os.path.join(dst_dir, CKPT)
    if os.path.isfile(dst) and os.path.getsize(dst) > 100e6:
        log("already there: %s" % dst)
        return dst
    log("downloading %s into %s" % (CKPT, dst_dir))
    tmp = dst + ".part"

    def hook(b, bs, total):
        if total > 0 and b % 200 == 0:
            print("  %3d%%" % min(100, b * bs * 100 // total), end="\r", flush=True)
    urllib.request.urlretrieve(CKPT_URL, tmp, hook)
    os.replace(tmp, dst)
    log("done: %s" % dst)
    return dst
