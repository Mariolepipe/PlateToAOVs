"""UI + worker process with fake models (offscreen): full run, the viewer following the steps,
and Cancel stopping a slow job right away.  QT_QPA_PLATFORM=offscreen python tests/test_ui_runner.py"""
import os, sys, tempfile, shutil, time, json, dataclasses
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE)); sys.path.insert(0, HERE)
from PySide6 import QtWidgets, QtCore
from laa import ui
from laa.config import Settings
import fakes

def start_fake(self):
    req = os.path.join(tempfile.gettempdir(), "laa_test_job.json")
    with open(req, "w") as f:
        json.dump({"settings": dataclasses.asdict(self.settings), "force": list(self.force), "until": self.until}, f)
    self.proc.start(sys.executable, ["-u", os.path.join(HERE, "fake_worker.py"), req])
ui.Runner.start = start_fake

app = QtWidgets.QApplication([])
ui.Main._load_last = lambda self: Settings()      # not the settings saved on this machine
ui.Main._missing_needed = lambda self, until=None: []    # fake models: nothing to download
QtWidgets.QMessageBox.question = staticmethod(lambda *a, **k: QtWidgets.QMessageBox.No)
tmp = tempfile.mkdtemp()
try:
    pattern = fakes.make_plate(os.path.join(tmp, "plate"))
    w = ui.Main()
    w.s = Settings(input_pattern=pattern, output_dir=os.path.join(tmp, "out"), crop_chunk=4, ltx_chunk=4, stab_radius=1, people_enabled=False, alpha_source="ltx")
    w._apply_quick(w.s); w._refresh_job(); w._set_page("aovs")
    t0 = time.time()                              # the plate loads by itself first
    while (w.runner is not None or not w._plate_ready()) and time.time() - t0 < 60:
        app.processEvents(); time.sleep(0.02)
    views, ends = [], []
    orig = w._set_view
    w._set_view = lambda k: (views.append(k), orig(k))
    w._finished_orig = w._finished
    def run_and_wait(timeout=120):
        ends.clear()
        w._run()
        w.runner.finished.connect(lambda e: ends.append(e))
        t0 = time.time()
        while not ends and time.time() - t0 < timeout:
            app.processEvents(); time.sleep(0.02)
        app.processEvents()
        return ends[0] if ends else "timeout"
    err = run_and_wait()
    print("full run:", repr(err), "| status:", w.status.text()[-40:])
    assert err == "", err
    print("viewer followed:", [v for i, v in enumerate(views) if i == 0 or views[i - 1] != v])
    assert {"final", "depth", "alpha"} <= set(views)          # default: only the deliverables
    assert not {"seg", "boxes", "moge", "merged"} & set(views), views
    assert w.more.isHidden() and w.seg.buttons["boxes"].isHidden()
    assert w.b_run.text() == "Up to date"
    # developer mode: the debug viewers come back and the viewer follows every step
    w.dev = True; w._set_page("aovs")
    assert not w.more.isHidden() and not w.seg.buttons["boxes"].isHidden()
    views.clear(); w.s.seg_size = "0.8b"; w.s.moge_refine_steps = 2; w._apply_quick(w.s); w._refresh_job()
    err = run_and_wait()
    print("developer mode, viewer followed:", [v for i, v in enumerate(views) if i == 0 or views[i - 1] != v])
    assert {"seg", "boxes", "moge", "merged", "final"} <= set(views), views
    w.dev = False; w._set_page("aovs")
    # cancel: slow fake segmentation (2 s / frame), cancel after 3 s
    os.environ["LAA_FAKE_SLOW"] = "2"
    w.s.seg_size = "0.4b"; w._apply_quick(w.s); w._refresh_job()
    ends.clear(); w._run(); w.runner.finished.connect(lambda e: ends.append(e))
    t0 = time.time()
    while time.time() - t0 < 3: app.processEvents(); time.sleep(0.02)
    print("status while slow:", w.status.text())
    tc = time.time(); w._cancel()
    while not ends and time.time() - tc < 10: app.processEvents(); time.sleep(0.01)
    print("cancel ->", ends, "in %.2fs" % (time.time() - tc))
    assert ends == ["cancelled"] and time.time() - tc < 2
    print("OK")
finally:
    shutil.rmtree(tmp, ignore_errors=True)
