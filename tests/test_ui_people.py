"""UI + worker (fake models incl. fake SAM3), offscreen: Generate stops on People, the panel lists
the numbers, a click adds the person behind, Apply re-runs SAM3 only, Continue runs the rest.
QT_QPA_PLATFORM=offscreen python tests/test_ui_people.py [shots_dir]"""
import os, sys, tempfile, shutil, time, json, dataclasses
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE)); sys.path.insert(0, HERE)
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
from PySide6 import QtWidgets, QtCore
from laa import ui
from laa.config import Settings
import fakes_people as FP
import numpy as np

SHOTS = sys.argv[1] if len(sys.argv) > 1 else None


def start_fake(self):
    req = os.path.join(tempfile.gettempdir(), "laa_test_job_people.json")
    with open(req, "w") as f:
        json.dump({"settings": dataclasses.asdict(self.settings), "force": list(self.force), "until": self.until}, f)
    self.proc.start(sys.executable, ["-u", os.path.join(HERE, "fake_worker_people.py"), req])
ui.Runner.start = start_fake
ui.SamServer.command = lambda self: (sys.executable, ["-u", os.path.join(HERE, "fake_sam_server.py")])

app = QtWidgets.QApplication([])
ui.Main._load_last = lambda self: Settings()      # not the settings saved on this machine
ui.Main._missing_needed = lambda self, until=None: []    # fake models: nothing to download
if SHOTS:                                   # same look as run.bat
    app.setStyle("Fusion"); app.setPalette(ui._dark_palette()); app.setStyleSheet(ui.STYLE)
QtWidgets.QMessageBox.question = staticmethod(lambda *a, **k: QtWidgets.QMessageBox.No)
tmp = tempfile.mkdtemp()


def shot(w, name):
    if SHOTS:
        os.makedirs(SHOTS, exist_ok=True)
        app.processEvents()
        w.grab().save(os.path.join(SHOTS, name))


try:
    pattern = FP.make_plate2(os.path.join(tmp, "plate"))
    w = ui.Main()
    w.resize(1400, 820)
    w.show()
    w.s = Settings(input_pattern=pattern, output_dir=os.path.join(tmp, "out"), crop_chunk=4, ltx_chunk=4,
                   stab_radius=0, depth_enabled=False)
    w._apply_quick(w.s); w._refresh_job()
    ends = []

    def wait(timeout=180):
        w.runner.finished.connect(lambda e: ends.append(e))
        t0 = time.time()
        while not ends and time.time() - t0 < timeout:
            app.processEvents(); time.sleep(0.02)
        for _ in range(5):
            app.processEvents(); time.sleep(0.02)
        return ends.pop() if ends else "timeout"

    t0 = time.time()                              # the plate (and depth) load by themselves
    while (w.runner is not None or not w._plate_ready()) and time.time() - t0 < 60:
        app.processEvents(); time.sleep(0.02)
    for _ in range(10):
        app.processEvents(); time.sleep(0.02)
    print("0. preload done | plate ready:", w._plate_ready(), "| depth fresh:", w.job.is_fresh("depth"),
          "| button:", w.b_run.text())
    assert w._plate_ready() and w.page == "people" and w.b_run.text() == "Find everyone"
    assert not w.seg.buttons["final"].isVisibleTo(w) and w.seg.buttons["people"].isVisibleTo(w)
    assert not w.people_box.isHidden()            # clicks possible before anything is found
    shot(w, "people_0_start.png")
    w._primary()
    err = wait()
    print("1. Find people ->", repr(err), "| button:", w.b_run.text(), "| view:", w.view)
    assert err in ("", "review") and w.b_run.text().startswith("Next") and w.view == "people"
    assert not w.people_box.isHidden()
    rows = [w.people_rows.itemAt(i).widget() for i in range(w.people_rows.count()) if w.people_rows.itemAt(i).widget()]
    print("   rows:", len(rows), "|", w.people_msg.text())
    assert len(rows) == 1
    shot(w, "people_1_review.png")

    # 2. fix: go to frame 3, new person, click on the one behind
    w.slider.setValue(3)
    w.b_fix.setChecked(True)
    w.fix_target.setCurrentIndex(w.fix_target.findData(0))
    a, b, _, _ = FP.two(3)
    ys, xs = np.where(b)
    t0 = time.time()                              # live SAM loads in its own process
    while (w.sam is None or not w.sam.is_ready) and time.time() - t0 < 60:
        app.processEvents(); time.sleep(0.02)
    assert w.sam is not None and w.sam.is_ready, "live SAM did not start"
    w._view_click(xs.mean() / FP.W, ys.mean() / FP.H, 1)
    t0 = time.time()
    while 3 not in w.previews and time.time() - t0 < 30:
        app.processEvents(); time.sleep(0.02)
    print("2. pending:", w.pending, "| hint:", w.fix_hint.text())
    assert list(w.pending) == [(3, 2)] and w.pending[(3, 2)]["points"][0][2] == 1
    from laa.io import sequence as seq
    pv = seq.read_png(w.previews[3])
    print("   live preview on frame 3, numbers:", sorted(set(np.unique(pv).tolist()) - {0}))
    assert sorted(set(np.unique(pv).tolist()) - {0}) == [1, 2]
    shot(w, "people_2_click.png")
    # undo the click -> preview gone, click again
    w._clear_clicks()
    assert not w.pending and 3 not in w.previews
    w._view_click(xs.mean() / FP.W, ys.mean() / FP.H, 1)
    w._apply_fixes()
    err = wait()
    rows = [w.people_rows.itemAt(i).widget() for i in range(w.people_rows.count()) if w.people_rows.itemAt(i).widget()]
    print("   after Apply ->", repr(err), "| rows:", len(rows), "| button:", w.b_run.text())
    assert err in ("", "review") and len(rows) == 2 and w.b_run.text().startswith("Next")
    shot(w, "people_3_fixed.png")

    # 3. untick person 2, tick again (decisions saved, no run)
    c2 = rows[1].findChild(QtWidgets.QCheckBox)
    c2.setChecked(False)
    assert w.job.people_edits()["removed"] == [2]
    c2.setChecked(True)
    assert w.job.people_edits()["removed"] == []

    # rename person 2 in its row -> saved, shown in the target list
    nm = rows[1].findChild(QtWidgets.QLineEdit)
    nm.setText("Back soldier"); nm.editingFinished.emit()
    for _ in range(5):
        app.processEvents(); time.sleep(0.02)
    assert w.job.people_edits()["names"] == {"2": "Back soldier"}
    assert w.fix_target.itemText(w.fix_target.findData(2)) == "Back soldier"
    shot(w, "people_3b_names.png")

    # a new object: click it on frame 3 -> pending row "Object 3" you can name right away
    w.b_fix.setChecked(True)
    w.fix_target.setCurrentIndex(w.fix_target.findData(-1))
    w._view_click(0.9, 0.2, 1)
    for _ in range(10):
        app.processEvents(); time.sleep(0.02)
    w._refresh_people()
    k_obj = w.fix_target.currentData()
    assert w.job.people_edits()["kinds"] == {str(k_obj): "object"}, w.job.people_edits()
    names = [r.findChild(QtWidgets.QLineEdit).placeholderText() for r in
             [w.people_rows.itemAt(i).widget() for i in range(w.people_rows.count())] if r]
    print("   new object ->", k_obj, "| rows:", names)
    assert "Object %d" % k_obj in names
    shot(w, "people_3c_object.png")
    w._clear_clicks()
    w.b_fix.setChecked(False)

    # 4. Next -> page 2, Generate -> everything else
    w._primary()
    print("4. Next -> page", w.page, "| button:", w.b_run.text(), "|", w.people_sum.text())
    assert w.page == "aovs" and w.b_run.text() == "Generate" and w.seg.buttons["final"].isVisibleTo(w)
    shot(w, "people_4_aovs.png")
    w._primary()
    err = wait()
    print("4. Continue ->", repr(err), "| status:", w.status.text()[-30:], "| button:", w.b_run.text())
    assert err == "" and w.b_run.text() == "Up to date"
    nums = w.job._read_done("alpha").get("numbers")
    assert nums == [1, 2], nums
    w._set_view("final")
    shot(w, "people_5_done.png")
    # Clear all: back to an empty selection
    w._set_page("people")
    QtWidgets.QMessageBox.question = staticmethod(lambda *a, **k: QtWidgets.QMessageBox.Yes)
    w._clear_all()
    for _ in range(10):
        app.processEvents(); time.sleep(0.02)
    e = w.job.people_edits()
    print("6. Clear all -> ready:", w._people_ready(), "| keys:", len(e["keys"]), "| button:", w.b_run.text())
    assert not w._people_ready() and not e["keys"] and not e.get("names") and w.b_run.text() == "Find everyone"
    shot(w, "people_6_cleared.png")
    print("OK")
finally:
    shutil.rmtree(tmp, ignore_errors=True)
