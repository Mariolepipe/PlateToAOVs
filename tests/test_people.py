"""People step (SAM 3.1 numbers + review) with fake models: two people, the one behind half hidden
by the one in front. The fake SAM3 first sees them as ONE person (the classic failure), the review
then adds the second one with a click. python tests/test_people.py"""
import os, sys, tempfile, shutil, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from fakes import *
from fakes_people import *

tmp = tempfile.mkdtemp()
try:
    folder = os.path.join(tmp, "plate")
    make_plate2(folder)
    from laa.config import Settings
    from laa.pipeline import Job
    from laa.stages import people as PP
    s = Settings(input_pattern=os.path.join(folder, "shot_020.####.png"), output_dir=os.path.join(tmp, "out"),
                 crop_chunk=4, ltx_chunk=4, stab_radius=0, depth_enabled=False, people_review=True)
    logs = []
    job = Job(s, log=logs.append)
    r = job.run()
    assert r == "review", (r, logs[-5:])
    ran = [l.split("]")[0][1:] for l in logs if "running" in l]
    print("1. first Generate stops for review after:", ran)
    assert ran == ["plate", "people"], ran                  # depth filter off by default
    ids = job.people_ids(2)
    print("   SAM3 alone numbers:", sorted(set(np.unique(ids)) - {0}))
    assert sorted(set(np.unique(ids)) - {0}) == [1]
    assert os.path.isfile(job.ppath("people", 2))

    # 2. review: on frame 3 click on the person behind -> "new person"
    a, b, _, _ = two(3)
    ys, xs = np.where(b)
    e = job.people_edits()
    e["keys"] = [{"frame": 3, "target": 2, "points": [[float(xs.mean()), float(ys.mean()), 1]], "base": None}]
    job.save_people_edits(e)
    logs.clear(); FakeSAM3.calls.clear(); FakeSAM2Video.calls.clear()
    r = Job(s, log=logs.append).run(until="people")
    print("2. keyframe -> propagate, SAM2 calls:", FakeSAM2Video.calls)
    assert ("key", 3, 2, False, 1) in FakeSAM2Video.calls and ("propagate",) in FakeSAM2Video.calls
    job = Job(s, log=logs.append)
    for i in range(N):
        a, b, _, _ = two(i)
        idi = job.people_ids(i)
        ok_a = (idi[a] == 1).mean()
        ok_b = (idi[b] == 2).mean()
        assert ok_a > 0.95 and ok_b > 0.95, (i, ok_a, ok_b)
    print("   every frame: person 1 = front, person 2 = behind (before and after the clicked frame)")

    # 3. Continue -> the rest runs, one LTX crop and one Sapiens crop per person
    e = job.people_edits(); e["validated"] = True; job.save_people_edits(e)
    logs.clear()
    r = Job(s, log=logs.append).run()
    assert r is None, logs[-3:]
    ran = [l.split("]")[0][1:] for l in logs if "running" in l]
    print("3. Continue runs:", ran)
    assert "people" not in ran and "write" in ran
    nums = Job(s, log=lambda *_: None)._read_done("alpha").get("numbers")
    print("   alpha (Selection, MatAnyone 2) per number:", nums)
    assert nums == [1, 2]
    bx = json.load(open(os.path.join(s.output_dir, "_cache", "crops", "boxes.json")))
    print("   Sapiens crops frame 0: %d (%d bodies)" % (len(bx["boxes"][0]), bx["n_body"][0]))
    assert bx["n_body"][0] == 2
    import OpenEXR
    ch = OpenEXR.File(job.exr_path(2), separate_channels=True).channels()
    assert "people.id" in ch and "alpha_1.A" in ch and "alpha_2.A" in ch, sorted(ch)
    a3e, b3e, _, _ = two(2)
    assert ch["alpha_2.A"].pixels[b3e & ~a3e].mean() > 0.9 and ch["alpha_2.A"].pixels[a3e].mean() < 0.1
    pid = ch["people.id"].pixels
    print("   EXR people.id values:", sorted(set(np.unique(pid).tolist())))
    # names: only the EXRs are rewritten, the header maps people.id -> name
    e = job.people_edits(); e["names"] = {"1": "Front soldier", "2": "Back soldier"}; job.save_people_edits(e)
    logs.clear()
    Job(s, log=logs.append).run()
    ran = [l.split("]")[0][1:] for l in logs if "running" in l]
    print("   rename ->", ran)
    assert ran == ["write"], ran
    hdr = OpenEXR.File(job.exr_path(2)).header()
    assert json.loads(hdr["people/names"]) == {"1": "Front soldier", "2": "Back soldier"}, hdr.get("people/names")

    # Find with two prompts: "person, car" -> one SAM 3.1 pass each, the car is numbered as an object
    from laa.models.sam3 import prompts, is_human
    assert prompts(" Person , car,") == ["person", "car"] and is_human("soldier") and not is_human("house")
    s3 = Settings(**{**s.__dict__}); s3.output_dir = os.path.join(tmp, "out_prompts"); s3.people_prompt = "person, car"
    j3 = Job(s3, log=lambda *_: None); FakeSAM3.calls.clear(); j3.run(until="people")
    d3 = Job(s3, log=lambda *_: None)._read_done("people")
    print("   Find 'person, car' -> SAM3 passes:", len(FakeSAM3.calls), "| found as:", d3.get("found_as"), "| kinds:", d3.get("kinds"))
    assert len(FakeSAM3.calls) == 2 and set(d3["found_as"].values()) <= {"person", "car"}
    assert all(d3["found_as"][k] == "car" for k in d3["kinds"])

    # objects: number 2 marked as an object -> alpha track kept, no Sapiens crop, listed in the header
    e = job.people_edits(); e["kinds"] = {"2": "object"}; job.save_people_edits(e)
    logs.clear()
    Job(s, log=logs.append).run()
    ran = [l.split("]")[0][1:] for l in logs if "running" in l]
    bx = json.load(open(os.path.join(s.output_dir, "_cache", "crops", "boxes.json")))
    nums = Job(s, log=lambda *_: None)._read_done("alpha").get("numbers")
    print("   2 = object ->", ran, "| body crops:", bx["n_body"][0], "| alpha numbers:", nums)
    assert "people" not in ran and "crops" in ran and bx["n_body"][0] == 1 and nums == [1, 2]
    jo = Job(s, log=lambda *_: None)
    a3 = jo.alpha(3); _, b3o, _, _ = two(3)
    print("   object pixels in the Sapiens (normals) mask: %.2f" % jo.person(3)[b3o].mean())
    assert jo.person(3)[b3o].mean() < 0.02
    # a rifle clicked as part of person 1 (in their SAM mask, not a body for Sapiens) stays in their
    # alpha mask even when there are objects elsewhere
    pid = seq.read_png(jo.cpath("people", 3, "png"))
    rifle = np.zeros_like(pid, bool); rifle[2:6, 2:20] = True
    pid2 = pid.copy(); pid2[rifle] = 1
    seq.write_png(jo.cpath("people", 3, "png"), pid2)
    jr = Job(s, log=lambda *_: None)
    H3, W3 = jr.plate(3).shape[:2]
    rifle_full = jr.people_ids_full(3, H3, W3) == 1
    rifle_full &= ~(jr.seg(3) > 0)
    held = jr.person(3, objects=True, with_objects=False)
    print("   held rifle in the person alpha mask: %.2f, in the normals mask: %.2f"
          % (held[rifle_full].mean(), jr.person(3)[rifle_full].mean()))
    assert held[rifle_full].mean() > 0.99 and jr.person(3)[rifle_full].mean() < 0.01
    seq.write_png(jo.cpath("people", 3, "png"), pid)
    print("   object alpha on its SAM mask: %.2f" % a3[b3o].mean())
    assert a3[b3o].mean() > 0.9
    assert OpenEXR.File(job.exr_path(2)).header()["people/objects"] == "2"
    # only objects (both numbers): no Sapiens step at all, MoGe normals everywhere, alpha = SAM masks
    e = job.people_edits(); e["kinds"] = {"1": "object", "2": "object"}; job.save_people_edits(e)
    logs.clear(); Job(s, log=logs.append).run()
    ran = [l.split("]")[0][1:] for l in logs if "running" in l]
    jo = Job(s, log=lambda *_: None)
    print("   only objects ->", ran, "| seg needed:", jo.needed("seg"), "| crops needed:", jo.needed("crops"))
    assert "seg" not in ran and "crops" not in ran and "merge" in ran and "alpha" in ran and "write" in ran
    a3 = jo.alpha(3); a, b, _, _ = two(3)
    assert a3[a].mean() > 0.9 and a3[b].mean() > 0.9
    assert "people.id" in OpenEXR.File(jo.exr_path(2), separate_channels=True).channels()
    e = job.people_edits(); e["kinds"] = {}; job.save_people_edits(e)
    logs.clear(); Job(s, log=logs.append).run()
    ran = [l.split("]")[0][1:] for l in logs if "running" in l]
    print("   people again ->", ran)
    assert "crops" in ran and "merge" in ran

    # 4. review change: remove person 2 -> no SAM3 rerun, crops / alpha redone, 1 crop left
    e = job.people_edits(); e["removed"] = [2]; job.save_people_edits(e)
    logs.clear(); FakeSAM3.calls.clear(); FakeSAM2Video.calls.clear()
    Job(s, log=logs.append).run()
    ran = [l.split("]")[0][1:] for l in logs if "running" in l]
    print("4. after removing #2:", ran, "| SAM3 calls:", len(FakeSAM3.calls))
    assert "people" not in ran and "crops" in ran and "alpha" in ran and not FakeSAM3.calls and not FakeSAM2Video.calls
    assert Job(s, log=lambda *_: None)._read_done("alpha").get("numbers") == [1]
    job = Job(s, log=logs.append)
    a, b, _, _ = two(3)
    import cv2
    b_in = cv2.erode(b.astype(np.uint8), np.ones((9, 9), np.uint8)) > 0     # the edge touches person 1
    assert not job.person(3)[b_in].any() and job.person(3)[a].mean() > 0.9
    # 5. "same as": 2 same as 1 -> one person
    e = job.people_edits(); e["removed"] = []; e["same"] = {"2": 1}; job.save_people_edits(e)
    assert sorted(set(np.unique(Job(s, log=logs.append).people_ids(3))) - {0}) == [1]
    lut = PP.lut({"removed": [3], "same": {"5": 4, "4": 1}})
    assert lut[5] == 1 and lut[3] == 0 and lut[2] == 2
    # 6. a right click takes someone out of a person: person 1 = both (SAM3), "not this" on the one behind
    j = Job(s, log=lambda *_: None)
    a3, b3, _, _ = two(3)
    os.makedirs(os.path.join(s.output_dir, "laa_people_keys"), exist_ok=True)
    seq.write_png(os.path.join(s.output_dir, "laa_people_keys", "f00003_p1.png"), ((a3 | b3) * 255).astype(np.uint8))
    e = j.people_edits()
    e["same"] = {}
    yb, xb = np.where(b3)
    e["keys"] = [{"frame": 3, "target": 1, "points": [[float(xb.mean()), float(yb.mean()), 0]],
                  "base": "laa_people_keys/f00003_p1.png"}]
    j.save_people_edits(e)
    Job(s, log=lambda *_: None).run(until="people")
    j = Job(s, log=lambda *_: None)
    print("6. right click on the one behind: person 1 on them %.2f, on the front one %.2f" % (
        (j.people_ids(3)[b3] == 1).mean(), (j.people_ids(3)[a3] == 1).mean()))
    assert (j.people_ids(3)[b3] == 1).mean() < 0.05 and (j.people_ids(3)[a3] == 1).mean() > 0.95
    assert (j.people_ids(6)[b3 & two(6)[1]] == 1).mean() < 0.05           # and through the shot
    # to_id_maps: the fixed person wins where they overlap
    PP.MIN_PX = 1
    mm = np.zeros((1, 2, 10, 10), bool); mm[0, 0, :, :5] = True; mm[0, 1, :, 4:] = True
    assert PP.to_id_maps(mm)[0, 0, 4] == 1 and PP.to_id_maps(mm, top=[1])[0, 0, 4] == 2
    # manual selection only (no automatic detection): click one person -> only them
    s2 = Settings(**{**s.__dict__}); s2.output_dir = os.path.join(tmp, "out_manual")
    j2 = Job(s2, log=lambda *_: None); j2.run(until="plate")
    a3, b3, _, _ = two(3)
    ys, xs = np.where(b3)
    e = j2.people_edits(); e["auto"] = False
    e["keys"] = [{"frame": 3, "target": 1, "points": [[float(xs.mean()), float(ys.mean()), 1]], "base": None}]
    j2.save_people_edits(e)
    Job(s2, log=lambda *_: None).run(until="people")
    j2 = Job(s2, log=lambda *_: None)
    got = sorted(set(np.unique(j2.people_ids(3))) - {0})
    print("   manual selection only ->", got, "| person 1 is the one behind:", (j2.people_ids(3)[b3] == 1).mean())
    assert got == [1] and (j2.people_ids(3)[b3] == 1).mean() > 0.95
    print("OK")
finally:
    shutil.rmtree(tmp, ignore_errors=True)
