"""Command line: python -m laa.cli --input D:/plate/plate.%04d.exr --colorspace acescg [--set key=value ...]"""

import argparse
import dataclasses
import json
import sys

from .config import Settings


def main(argv=None):
    ap = argparse.ArgumentParser(description="LiveActionAOVs: N / depth / P / alpha passes for Nuke, no ComfyUI server.")
    ap.add_argument("--input", required=False, help="sequence: folder, plate.%%04d.exr, plate.####.exr or one frame")
    ap.add_argument("--output", default="", help="output folder (default: aovs_<name> next to the plate)")
    ap.add_argument("--colorspace", default=None, choices=["srgb", "linear", "acescg"])
    ap.add_argument("--first", type=int, default=None)
    ap.add_argument("--last", type=int, default=None)
    ap.add_argument("--job", default="", help="laa_job.json of a previous run (settings)")
    ap.add_argument("--until", default=None, help="stop after this step")
    ap.add_argument("--force", default="", help="comma separated steps to re-run")
    ap.add_argument("--set", action="append", default=[], help="any setting, e.g. --set alpha_source=matting")
    ap.add_argument("--list-settings", action="store_true")
    a = ap.parse_args(argv)

    s = Settings()
    if a.list_settings:
        print(s.to_json())
        return 0
    if a.job:
        with open(a.job) as f:
            s = Settings.from_dict(json.load(f))
    if a.input:
        s.input_pattern = a.input
    if a.output:
        s.output_dir = a.output
    if a.colorspace:
        s.colorspace = a.colorspace
    if a.first is not None:
        s.first = a.first
    if a.last is not None:
        s.last = a.last
    types = {f.name: f.type for f in dataclasses.fields(Settings)}
    for kv in a.set:
        k, v = kv.split("=", 1)
        t = types.get(k)
        if t is None:
            ap.error("unknown setting %s" % k)
        t = t if isinstance(t, type) else {"bool": bool, "int": int, "float": float, "str": str}[t]
        setattr(s, k, (v.lower() in ("1", "true", "yes", "on")) if t is bool else t(v))
    if not s.input_pattern:
        ap.error("--input is required")

    from .pipeline import Job
    job = Job(s, progress=lambda st, d, t: print("\r  %s %d/%d" % (st, d, t), end="", flush=True))
    job.log = lambda m: print("\n" + m if m.startswith("[") else m, flush=True)
    job.run(until=a.until, force=tuple(x for x in a.force.split(",") if x))
    print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
