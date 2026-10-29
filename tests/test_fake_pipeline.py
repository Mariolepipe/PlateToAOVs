"""Runs the whole pipeline with fake models (no weights, CPU): checks the plumbing, the cache,
the EXR layout and that CHECK A is ~0 on people. python tests/test_fake_pipeline.py"""
import os, sys, tempfile, shutil, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, torch
from fakes import *

tmp = tempfile.mkdtemp()
try:
    # single-channel PNGs must come back unchanged (they used to be written mirrored)
    m = np.zeros((5, 9), np.uint8); m[:, :2] = 7
    seq.write_png(os.path.join(tmp, "m.png"), m)
    assert (seq.read_png(os.path.join(tmp, "m.png")) == m).all(), "mask mirrored"
    a16 = (np.arange(9, dtype=np.uint16) * 1000)[None].repeat(3, 0)
    seq.write_png(os.path.join(tmp, "a.png"), a16)
    assert (seq.read_png(os.path.join(tmp, "a.png")) == a16).all(), "alpha mirrored"
    pattern = make_plate(os.path.join(tmp, "plate"))
    from laa.config import Settings
    from laa.pipeline import Job
    s = Settings(input_pattern=pattern, output_dir=os.path.join(tmp, "out"), alpha_source="ltx",
                 crop_chunk=4, ltx_chunk=4, stab_radius=1, comparison_layers=True, people_enabled=False)
    logs = []
    job = Job(s, log=logs.append)
    job.run()
    print("\n".join(l for l in logs if "CHECK" in l or "signs" in l))
    exr = job.exr_path(0)
    assert os.path.isfile(exr), exr
    import cv2
    print("EXR files:", sorted(f for f in os.listdir(s.output_dir) if f.endswith(".exr"))[:3], "...")
    # read back the EXR header channel list
    raw = open(exr, "rb").read(4000)
    for c in (b"N.R", b"P.B", b"depth.Z", b"A\x00", b"Nmoge.R", b"moge_valid.A"):
        assert c in raw, c
    # LTX alpha on the whole frame (Foreground): ~1 on people, 0 far from them
    a5 = job.alpha(3); seg5 = job.seg(3) > 0
    print("alpha on people %.2f | far from people %.3f" % (a5[seg5].mean(), a5[:, :40].mean()))
    assert a5[seg5].mean() > 0.8 and a5[:, :40].mean() < 0.01
    # cache: second run must skip everything
    logs.clear(); job2 = Job(s, log=logs.append); job2.run()
    assert sum("up to date" in l for l in logs) == 9, logs      # albedo is off: "not needed"
    # change padding: crops, merge, stabilize, write re-run; moge / ltx alpha / depth stay
    logs.clear(); s.crop_padding = 0.12; Job(s, log=logs.append).run()
    ran = [l.split("]")[0][1:] for l in logs if "running" in l]
    print("re-ran after padding change:", ran)
    assert ran == ["crops", "merge", "stabilize", "write"], ran
    # matting alpha
    logs.clear(); s.alpha_source = "matting"; Job(s, log=logs.append).run()
    ran = [l.split("]")[0][1:] for l in logs if "running" in l]
    print("re-ran after alpha=matting:", ran)
    logs.clear(); Job(s, log=logs.append).run(force=("crops",))
    ran = [l.split("]")[0][1:] for l in logs if "running" in l]
    print("re-ran after forcing crops:", ran)
    assert ran == ["crops", "merge", "stabilize", "alpha", "write"], ran
    # albedo only with Cosmos (fake): letterbox in, un-letterbox out, layers in the EXR
    s4 = Settings(**{**s.__dict__}); s4.normals_enabled = s4.alpha_enabled = s4.depth_enabled = False
    s4.albedo_enabled = True; s4.albedo_source = "cosmos"; s4.cosmos_material = True; s4.cosmos_face_crops = True
    s4.output_dir = os.path.join(tmp, "out_cosmos")
    j4 = Job(s4, log=logs.append); j4.run()
    import OpenEXR
    ch = OpenEXR.File(j4.exr_path(2), separate_channels=True).channels()
    print("cosmos channels:", sorted(ch))
    assert sorted(ch) == ["albedo.B", "albedo.G", "albedo.R", "metallic.Y", "roughness.Y"], sorted(ch)
    alb = ch["albedo.R"].pixels.astype(np.float32)
    assert alb.shape == (H, W), alb.shape
    plate = j4.plate(2).astype(np.float32) / 255.0
    lin = np.where(plate <= 0.04045, plate / 12.92, ((plate + 0.055) / 1.055) ** 2.4)[..., 0]
    err = np.abs(alb - lin)[5:-5, 5:-5].mean()
    print("  albedo vs plate (identity fake) mean err %.4f" % err)
    assert err < 0.02, err
    done = j4._read_done("albedo")
    print("  faces refined:", done.get("faces"), "| steps run:", [l.split("]")[0][1:] for l in logs if "running" in l][-3:])
    assert done.get("faces", 0) >= 1, done
    assert os.path.isfile(j4.ppath("albedo_faces", 2)) and os.path.isfile(j4.ppath("albedo_full", 2))
    # faces off: no segmentation needed any more
    s5 = Settings(**{**s4.__dict__}); s5.cosmos_face_crops = False; s5.output_dir = os.path.join(tmp, "out_cosmos2")
    logs.clear(); Job(s5, log=logs.append).run()
    ran = [l.split("]")[0][1:] for l in logs if "running" in l]
    print("  faces off:", ran); assert ran == ["plate", "albedo", "write"], ran
    # albedo only: nothing else runs, the EXR only has albedo layers
    s2 = Settings(**{**s.__dict__}); s2.normals_enabled = s2.alpha_enabled = s2.depth_enabled = False
    s2.albedo_enabled = True; s2.albedo_source = "marigold"; s2.output_dir = os.path.join(tmp, "out_albedo")
    logs.clear(); Job(s2, log=logs.append).run()
    ran = [l.split("]")[0][1:] for l in logs if "running" in l]
    print("albedo only:", ran)
    assert ran == ["plate", "albedo", "write"], ran
    import OpenEXR
    chs = sorted(OpenEXR.File(Job(s2).exr_path(0), separate_channels=True).channels())
    print("  channels:", chs)
    assert chs == ["albedo.B", "albedo.G", "albedo.R", "roughness.Y"], chs
    # nothing on: refused
    s3 = Settings(**{**s2.__dict__}); s3.albedo_enabled = False
    try:
        Job(s3, log=logs.append).run(); raise AssertionError("should refuse")
    except ValueError as e:
        print("  nothing on ->", e)
    print("previews:", sorted(os.listdir(os.path.join(s.output_dir, "_preview"))))
    # check values: N inside person = flipped sapiens (= moge direction after signs) -> facing camera +Z
    print("OK")
finally:
    if os.environ.get("KEEP"):
        print("kept", tmp)
    else:
        shutil.rmtree(tmp)
