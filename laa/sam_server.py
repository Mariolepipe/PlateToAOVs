"""
Live clicks on the People page: SAM 2.1 stays loaded in this process while you click, each click
re-cuts the frame in a fraction of a second (python -m laa.sam_server job.json, started by the UI).

stdin : one JSON command per line
          {"cmd": "preview", "id": 7, "frame": 12, "keys": [ ...the keyframes of that frame... ]}
          {"cmd": "quit"}
stdout: "@@LAA {json}" lines, same as the worker
          {"t": "ready"} | {"t": "preview", "id": 7, "frame": 12, "path": ".../preview_00012_3.png"}
          {"t": "error", "id": 7, "m": "..."}
The preview is the frame's id image: the people of the last propagation, with the people you are
clicking on re-cut on top. The same replay as in the shot run, so what you see is what you get.
"""

import json
import os
import sys

TAG = "@@LAA "
_OUT = sys.stdout


def send(**msg):
    _OUT.write(TAG + json.dumps(msg) + "\n")
    _OUT.flush()


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    with open(argv[0], "r", encoding="utf-8") as f:
        req = json.load(f)
    try:
        import numpy as np
        from .config import Settings
        from . import pipeline as P
        from .io import sequence as seq
        from .stages import people as PP
        from .models import sam2 as S2
        from .models.sam3 import work_size
        s = Settings.from_dict(req["settings"])
        job = P.Job(s, log=lambda m: send(t="log", m=m))
        H, W = job.plate(0).shape[:2]
        w, h = work_size(W, H)
        people_fresh = job.is_fresh("people")
        frames_dir = job.people_frames_dir()
        v = S2.SAM2Video(frames_dir, log=lambda m: send(t="log", m=m))
        out_dir = os.path.join(job.cache, "people_preview")
        os.makedirs(out_dir, exist_ok=True)
    except Exception as e:
        import traceback
        send(t="log", m=traceback.format_exc())
        send(t="error", id=-1, m=str(e) or e.__class__.__name__)
        return 1
    send(t="ready")
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            cmd = json.loads(line)
        except ValueError:
            continue
        if cmd.get("cmd") == "quit":
            break
        if cmd.get("cmd") != "preview":
            continue
        rid = cmd.get("id", 0)
        try:
            i = int(cmd["frame"])
            idp = job.cpath("people", i, "png")
            ids = seq.read_png(idp) if people_fresh and os.path.isfile(idp) else np.zeros((h, w), np.uint8)
            for key in cmd.get("keys", []):
                k = int(key["target"])
                m = v.set_key(i, k, job.key_base(key, h, w), PP.scaled_points(key, w / float(W)))
                ids[ids == k] = 0
                if m is not None:
                    ids[m] = k
            p = os.path.join(out_dir, "preview_%05d_%d.png" % (i, rid % 4))
            seq.write_png(p, ids)
            send(t="preview", id=rid, frame=i, path=p)
        except Exception as e:
            import traceback
            send(t="log", m=traceback.format_exc())
            send(t="error", id=rid, m=str(e) or e.__class__.__name__)
    try:
        v.release()
    except Exception:
        pass
    return 0


if __name__ == "__main__":
    sys.exit(main())
