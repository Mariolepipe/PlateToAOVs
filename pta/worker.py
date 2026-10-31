"""
The job runs in its own process (python -m pta.worker job.json), started by the UI.
Cancel = the UI kills this process: instant, even in the middle of loading a 21 GB model or a
diffusion step, and Windows frees all its RAM / VRAM at once. Frames already written to the cache
are kept (a step is only marked done when it finishes).

Messages to the UI are single lines on stdout: "@@PTA {json}". Anything else printed (model
libraries) is just log text.
"""

import json
import sys

TAG = "@@PTA "
_OUT = sys.stdout          # kept: model code may redirect sys.stdout to silence its own prints


def send(**msg):
    _OUT.write(TAG + json.dumps(msg) + "\n")
    _OUT.flush()


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    with open(argv[0], "r", encoding="utf-8") as f:
        req = json.load(f)
    try:
        from .config import Settings
        from . import pipeline as P
        job = P.Job(Settings.from_dict(req["settings"]),
                    log=lambda m: send(t="log", m=m),
                    progress=lambda step, d, n: send(t="progress", step=step, d=d, n=n),
                    detail=lambda text: send(t="detail", m=text))
        r = job.run(until=req.get("until") or None, force=tuple(req.get("force", ())))
        send(t="finished", err="review" if r == "review" else "")
        return 0
    except Exception as e:
        import traceback
        send(t="log", m=traceback.format_exc())
        err = str(e) or e.__class__.__name__
        try:
            from . import pipeline as P
            if P.disk_full(e) and not err.startswith("Disk full"):
                err = P.disk_full_text(job.out)
        except Exception:
            pass
        send(t="finished", err=err)
        return 1


if __name__ == "__main__":
    sys.exit(main())
