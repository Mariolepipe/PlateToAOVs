"""
MoGe speed test: the same frames with torch.inference_mode() (the fix) and without (autograd on,
the old behaviour). Each mode runs in its own Python process, so a crash in the old mode can't
spoil the other one, and both start with an empty GPU.

    python -m laa.bench_moge "C:/shots/my_shot/aovs_my_shot" --frames 12

Uses the plate already in that output folder (_cache/plate) and the MoGe settings of its
laa_job.json. Frame 1 includes warm-up, so the average leaves it out.
"""

import argparse
import json
import os
import subprocess
import sys
import time

TAG = "@@BENCH "


def child(out_dir, frames, mode):
    import torch
    from .config import Settings
    from .io import sequence as seq
    from .models.moge import MoGe

    with open(os.path.join(out_dir, "laa_job.json"), "r", encoding="utf-8") as f:
        s = Settings.from_dict(json.load(f))
    plate_dir = os.path.join(out_dir, "_cache", "plate")
    files = sorted(p for p in os.listdir(plate_dir) if p.endswith(".png"))[:frames]
    if not files:
        raise SystemExit("no plate frames in %s (run the plate step first)" % plate_dir)

    def say(**m):
        sys.stdout.write(TAG + json.dumps(m) + "\n"); sys.stdout.flush()

    t0 = time.perf_counter()
    m = MoGe(s.moge_model, log=lambda *_: None)
    say(t="load", s=time.perf_counter() - t0)
    for k, name in enumerate(files):
        rgb = seq.read_png(os.path.join(plate_dir, name))
        torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats()
        t = time.perf_counter()
        try:
            if mode == "inference":
                m.infer(rgb, s.moge_resolution_level, s.fov_x, s.moge_refine_steps)
            else:                                   # the old way: no inference_mode, autograd records everything
                img = torch.from_numpy(rgb).float().div(255.0).permute(2, 0, 1)[None]
                r = m._infer(img, s.moge_resolution_level, s.fov_x, s.moge_refine_steps)
                {k2: v[0].detach().float().cpu().numpy() for k2, v in r.items()
                 if k2 in ("normal", "depth") and v is not None}
                del r
            torch.cuda.synchronize()
        except Exception as e:
            say(t="crash", frame=k + 1, err=str(e).splitlines()[0])
            return
        say(t="frame", frame=k + 1, s=time.perf_counter() - t,
            peak_gb=torch.cuda.max_memory_allocated() / 2 ** 30)
    say(t="end")


def run_mode(out_dir, frames, mode):
    print("\n== %s ==" % ("WITH inference_mode (fix)" if mode == "inference" else "WITHOUT (old, autograd on)"))
    p = subprocess.Popen([sys.executable, "-u", "-m", "laa.bench_moge", out_dir, "--frames", str(frames),
                          "--child", mode], stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                         text=True, encoding="utf-8", errors="replace")
    res = {"load": None, "times": [], "peaks": [], "crash": None}
    other = []
    for line in p.stdout:
        if not line.startswith(TAG):
            other.append(line.rstrip())
            continue
        m = json.loads(line[len(TAG):])
        if m["t"] == "load":
            res["load"] = m["s"]; print("  model load   %.1f s" % m["s"])
        elif m["t"] == "frame":
            res["times"].append(m["s"]); res["peaks"].append(m["peak_gb"])
            print("  frame %3d    %6.2f s    peak VRAM %5.1f GB" % (m["frame"], m["s"], m["peak_gb"]))
        elif m["t"] == "crash":
            res["crash"] = (m["frame"], m["err"])
            print("  frame %3d    CRASH: %s" % (m["frame"], m["err"]))
    p.wait()
    if not res["times"] and not res["crash"]:          # died before the first frame: show why
        print("\n".join("  | " + l for l in other[-20:]))
    return res


def avg(xs):
    xs = xs[1:] if len(xs) > 1 else xs              # frame 1 = warm-up
    return sum(xs) / len(xs) if xs else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out_dir")
    ap.add_argument("--frames", type=int, default=12)
    ap.add_argument("--child")
    a = ap.parse_args()
    if a.child:
        return child(a.out_dir, a.frames, a.child)

    print("Watch Task Manager > Performance > GPU > 'Shared GPU memory' during the first (old) run.")
    old = run_mode(a.out_dir, a.frames, "grad")          # the heavy one first
    new = run_mode(a.out_dir, a.frames, "inference")

    print("\n================ RESULT ================")
    for name, r in (("with inference_mode", new), ("without (old)", old)):
        t = avg(r["times"])
        line = "%-22s" % name
        line += ("%6.2f s/frame" % t) if t else "     -       "
        line += ("   peak %5.1f GB" % max(r["peaks"])) if r["peaks"] else ""
        line += ("   crashed at frame %d" % r["crash"][0]) if r["crash"] else "   no crash (%d frames)" % len(r["times"])
        print(line)
    tn, to = avg(new["times"]), avg(old["times"])
    if tn and to:
        print("\n-> %.1fx faster per frame, %.1f GB less peak VRAM"
              % (to / tn, max(old["peaks"]) - max(new["peaks"])))


if __name__ == "__main__":
    main()
