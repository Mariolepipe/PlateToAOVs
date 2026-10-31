"""MoGe frames start..n in a process of their own (python -m pta.moge_child req.json), started by
Job.step_moge. If the CUDA driver gives up mid-shot, only this process dies: the frames written so far
stay in the cache and step_moge starts a new one from the next frame. Same "@@PTA {json}" lines as
pta.worker."""

import json
import sys

from .worker import send


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
        job._moge_frames(int(req["start"]), int(req["level"]), int(req["steps"]))
        return 0
    except Exception as e:
        import traceback
        send(t="log", m=traceback.format_exc())
        from .pipeline import disk_full
        return 3 if disk_full(e) else 1           # 3: disk full, step_moge stops instead of retrying


if __name__ == "__main__":
    sys.exit(main())
