"""
PlateToAOVs desktop app (PySide6).

One window, three things: pick a plate, look at it, Generate. Everything technical (model sizes,
crop padding, stabilization...) lives in the Advanced settings sheet; the pipeline's inner steps
are under the viewer's "More" menu for when something needs checking.
"""

import dataclasses
import json
import os
import sys
import tempfile
import time

import numpy as np
from PySide6 import QtCore, QtGui, QtWidgets

from .config import Settings, APP_DIR as _APP
from . import pipeline as P

APP_DIR = str(_APP)

ACCENT = "#0a84ff"          # macOS dark-mode blue
# the app runs in pythonw.exe (no black window); its jobs run in python.exe (they print their
# progress on stdout), without a window either: QProcess hides it when the app has no console
_d, _b = os.path.split(sys.executable)
PYTHON = os.path.join(_d, "python.exe") if _b.lower() == "pythonw.exe" and os.path.isfile(os.path.join(_d, "python.exe")) \
    else sys.executable

ASSETS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets")
STYLE = """
* { font-family: 'Segoe UI Variable Text', 'Segoe UI', 'SF Pro Text', sans-serif; font-size: 13px; color: #ececec; }
QMainWindow, QDialog { background: #1c1c1e; }
QToolTip { background: #2c2c2e; color: #ececec; border: 1px solid #3a3a3c; padding: 4px 6px; }
QWidget#side { background: #232325; border-left: 1px solid #2f2f31; }
QLabel#h1 { font-size: 20px; font-weight: 600; }
QLabel#h2 { font-size: 12px; font-weight: 600; color: #8e8e93; }
QLabel#muted { color: #8e8e93; }
QLabel#plate { font-size: 15px; font-weight: 600; }
QLineEdit, QSpinBox, QDoubleSpinBox, QComboBox {
    background: #2c2c2e; border: 1px solid #3a3a3c; border-radius: 7px; padding: 5px 8px; min-height: 18px; }
QLineEdit:focus, QSpinBox:focus, QDoubleSpinBox:focus, QComboBox:focus { border-color: %(a)s; }
QComboBox::drop-down { border: none; width: 18px; }
QPushButton { background: #3a3a3c; border: none; border-radius: 7px; padding: 6px 14px; }
QPushButton:hover { background: #48484a; }
QPushButton:disabled { color: #636366; }
QPushButton:checked { background: %(a)s; color: white; }
QPushButton#primary { background: %(a)s; color: white; font-weight: 600; font-size: 14px; padding: 10px; border-radius: 9px; }
QPushButton#primary:hover { background: #339dff; }
QPushButton#primary:disabled { background: #1f3a5c; color: #8e8e93; }
QPushButton#link { background: transparent; color: %(a)s; padding: 2px 0; text-align: left; }
QPushButton#link:hover { color: #5eb0ff; }
QToolButton { background: transparent; border: none; border-radius: 6px; padding: 5px 12px; color: #c7c7cc; }
QToolButton:hover { background: #2c2c2e; }
QToolButton:checked { background: #3a3a3c; color: white; }
QToolButton#play { background: transparent; border: none; border-radius: 6px; padding: 0; min-width: 30px; max-width: 30px; min-height: 30px; max-height: 30px; }
QToolButton#play:hover { background: #2c2c2e; }
QToolButton#play:checked { background: transparent; }
QToolButton#play:checked:hover { background: #2c2c2e; }
QWidget#segbar { background: #232325; border-radius: 8px; }
QProgressBar { background: #3a3a3c; border: none; border-radius: 3px; max-height: 6px; }
QProgressBar::chunk { background: %(a)s; border-radius: 3px; }
QSlider::groove:horizontal { height: 4px; background: #3a3a3c; border-radius: 2px; }
QSlider::sub-page:horizontal { background: %(a)s; border-radius: 2px; }
QSlider::handle:horizontal { background: white; width: 14px; height: 14px; margin: -5px 0; border-radius: 7px; }
QCheckBox { spacing: 9px; }
QCheckBox::indicator { width: 16px; height: 16px; border-radius: 4px; border: 1px solid #545456; background: #2c2c2e; }
QCheckBox::indicator:checked { background: %(a)s; border-color: %(a)s; image: url(%(check)s); }
QCheckBox::indicator:checked:disabled { background: #3a3a3c; border-color: #3a3a3c; }
QCheckBox:disabled { color: #636366; }
QComboBox:disabled { color: #636366; }
QAbstractSpinBox::up-button, QAbstractSpinBox::down-button { width: 0; border: none; }
QPlainTextEdit { background: #161618; border: none; border-radius: 7px; color: #a1a1a6; font-family: Consolas, monospace; font-size: 11px; }
QScrollArea { border: none; background: #1c1c1e; }
QScrollArea > QWidget > QWidget, QWidget#advhost { background: #1c1c1e; }
QDialog QLabel { color: #ececec; }
QScrollBar:vertical { background: #1c1c1e; width: 10px; margin: 0; }
QScrollBar::handle:vertical { background: #48484a; border-radius: 5px; min-height: 30px; }
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical { height: 0; }
QScrollBar::add-page:vertical, QScrollBar::sub-page:vertical { background: none; }
QMenu { background: #2c2c2e; border: 1px solid #3a3a3c; border-radius: 8px; padding: 4px; }
QMenu::item { padding: 6px 18px; border-radius: 5px; }
QMenu::item:selected { background: %(a)s; }
""" % {"a": ACCENT, "check": os.path.join(os.path.dirname(os.path.abspath(__file__)), "check.svg").replace("\\", "/")}

# what the viewer shows. Main row = what you deliver; "More" = the inner steps, for checking.
MAIN_VIEWS = [("plate", "Plate"), ("people", "Segmentation"), ("boxes", "Crops"), ("final", "Normals"),
              ("depth", "Depth"), ("alpha", "Alpha"), ("albedo", "Albedo")]
MORE_VIEWS = [("seg", "Body parts (Sapiens)"), ("tiles", "Crops in detail"), ("sapiens", "Sapiens normals"), ("moge", "MoGe normals"),
              ("blend", "Blend mask"), ("merged", "Before stabilization"), None,
              ("checkA", "Check: crops pasted correctly"), ("checkB", "Check: stabilization"), None,
              ("alpha_tiles", "Alpha crops (Fast)"), None,
              ("albedo_full", "Albedo before face crops"), ("albedo_faces", "Albedo face crops")]
VIEW_NAMES = dict(MAIN_VIEWS + [v for v in MORE_VIEWS if v])
# shown only in developer mode (Advanced > Developer): the inner steps, for checking the pipeline
DEV_MAIN_VIEWS = {"boxes"}
DEV_FIELDS = {"cosmos_face_crops", "cosmos_face_size", "cosmos_face_edge", "comparison_layers"}

# plain-words status for each step
STEP_TEXT = {"plate": "Reading the plate", "people": "Finding and numbering people", "seg": "Body parts", "crops": "Normals on people",
             "moge": "Normals and camera of the scene", "depth": "Depth", "alpha": "Alpha",
             "albedo": "Albedo", "merge": "Combining normals", "stabilize": "Removing flicker",
             "write": "Writing EXRs"}
STEP_VIEW = {"plate": "plate", "people": "people", "seg": "seg", "crops": "boxes", "moge": "moge", "depth": "depth",
             "alpha": "alpha", "albedo": "albedo", "merge": "merged", "stabilize": "final"}


def _icon(kind, color="#ececec", size=36):
    """drawn play / pause icons (crisp at any DPI, unlike text glyphs)."""
    pm = QtGui.QPixmap(size * 2, size * 2)
    pm.fill(QtCore.Qt.transparent)
    p = QtGui.QPainter(pm)
    p.setRenderHint(QtGui.QPainter.Antialiasing)
    s = size * 2
    p.setBrush(QtGui.QColor(color))
    if kind == "play":                      # filled triangle with rounded corners
        pen = QtGui.QPen(QtGui.QColor(color), s * 0.07)
        pen.setJoinStyle(QtCore.Qt.RoundJoin)
        p.setPen(pen)
        path = QtGui.QPainterPath()
        path.moveTo(s * 0.38, s * 0.30)
        path.lineTo(s * 0.71, s * 0.5)
        path.lineTo(s * 0.38, s * 0.70)
        path.closeSubpath()
        p.drawPath(path)
        p.end()
        return QtGui.QIcon(pm)
    p.setPen(QtCore.Qt.NoPen)                # pause: two rounded bars
    w = s * 0.12
    p.drawRoundedRect(QtCore.QRectF(s * 0.32, s * 0.29, w, s * 0.42), w * 0.35, w * 0.35)
    p.drawRoundedRect(QtCore.QRectF(s * 0.56, s * 0.29, w, s * 0.42), w * 0.35, w * 0.35)
    p.end()
    return QtGui.QIcon(pm)


# ------------------------------------------------------------------------------------- viewer
class Viewer(QtWidgets.QGraphicsView):
    def __init__(self):
        super().__init__()
        self.setScene(QtWidgets.QGraphicsScene(self))
        self.item = QtWidgets.QGraphicsPixmapItem()
        self.item.setTransformationMode(QtCore.Qt.SmoothTransformation)
        self.scene().addItem(self.item)
        self.setBackgroundBrush(QtGui.QColor("#121214"))
        self.setDragMode(QtWidgets.QGraphicsView.ScrollHandDrag)
        self.setTransformationAnchor(QtWidgets.QGraphicsView.AnchorUnderMouse)
        self.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
        self.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
        self.setFrameShape(QtWidgets.QFrame.NoFrame)
        self._fit = True
        self.msg = ""
        self.click_mode = False

    clicked = QtCore.Signal(float, float, int)        # image x, y (fraction of the width / height), 1 left 2 right

    def set_click_mode(self, on):
        self.click_mode = bool(on)
        self.setDragMode(QtWidgets.QGraphicsView.NoDrag if on else QtWidgets.QGraphicsView.ScrollHandDrag)
        self.viewport().setCursor(QtCore.Qt.CrossCursor if on else QtCore.Qt.ArrowCursor)

    def mousePressEvent(self, e):
        if self.click_mode and not self.item.pixmap().isNull() and e.button() in (QtCore.Qt.LeftButton, QtCore.Qt.RightButton):
            p = self.mapToScene(e.position().toPoint())
            pm = self.item.pixmap()
            fx, fy = p.x() / pm.width(), p.y() / pm.height()
            if 0 <= fx <= 1 and 0 <= fy <= 1:
                self.clicked.emit(fx, fy, 1 if e.button() == QtCore.Qt.LeftButton else 2)
            return
        super().mousePressEvent(e)

    def show_array(self, rgb):
        h, w = rgb.shape[:2]
        img = QtGui.QImage(rgb.data, w, h, 3 * w, QtGui.QImage.Format_RGB888).copy()
        self._set_pixmap(QtGui.QPixmap.fromImage(img), "")

    def show_image(self, path, msg=""):
        pm = QtGui.QPixmap(path) if path and os.path.isfile(path) else QtGui.QPixmap()
        self._set_pixmap(pm, msg)

    def _set_pixmap(self, pm, msg):
        self.item.setPixmap(pm)
        self.msg = "" if not pm.isNull() else msg
        if not pm.isNull():
            self.scene().setSceneRect(self.item.boundingRect())
            if self._fit:
                self.fit()
        self.viewport().update()

    def fit(self):
        if not self.item.pixmap().isNull():
            self.fitInView(self.item, QtCore.Qt.KeepAspectRatio)
            self._fit = True

    def wheelEvent(self, e):
        f = 1.15 if e.angleDelta().y() > 0 else 1 / 1.15
        self.scale(f, f)
        self._fit = False

    def mouseDoubleClickEvent(self, e):
        self.fit()

    def resizeEvent(self, e):
        super().resizeEvent(e)
        if self._fit:
            self.fit()

    def drawForeground(self, painter, rect):
        if not self.msg:
            return
        painter.save()
        painter.resetTransform()
        painter.setPen(QtGui.QColor("#636366"))
        f = painter.font()
        f.setPointSize(12)
        painter.setFont(f)
        painter.drawText(self.viewport().rect(), QtCore.Qt.AlignCenter, self.msg)
        painter.restore()


# ------------------------------------------------------------------------------------- worker
class Runner(QtCore.QObject):
    """Runs the job in a separate Python process (pta.worker). Cancel kills it: immediate."""
    log = QtCore.Signal(str)
    progress = QtCore.Signal(str, int, int)
    detail = QtCore.Signal(str)
    finished = QtCore.Signal(str)

    def __init__(self, settings, force=(), parent=None, until=None):
        super().__init__(parent)
        self.settings, self.force, self.until = settings, force, until
        self.cancelled = False
        self._buf = b""
        self._err = None
        self.proc = QtCore.QProcess(self)
        self.proc.setProcessChannelMode(QtCore.QProcess.MergedChannels)
        env = QtCore.QProcessEnvironment.systemEnvironment()
        env.insert("PYTHONIOENCODING", "utf-8")
        env.insert("PYTHONUNBUFFERED", "1")
        self.proc.setProcessEnvironment(env)
        self.proc.setWorkingDirectory(APP_DIR)
        self.proc.readyReadStandardOutput.connect(self._read)
        self.proc.finished.connect(self._done)

    def start(self):
        req = os.path.join(tempfile.gettempdir(), "pta_job_%d.json" % os.getpid())
        with open(req, "w", encoding="utf-8") as f:
            json.dump({"settings": dataclasses.asdict(self.settings), "force": list(self.force),
                       "until": self.until}, f)
        self.proc.start(PYTHON, ["-u", "-m", "pta.worker", req])

    def cancel(self):
        self.cancelled = True
        if sys.platform == "win32" and self.proc.processId():
            # whole tree: the worker and anything it started (wsl.exe for Cosmos)
            import subprocess
            subprocess.Popen(["taskkill", "/F", "/T", "/PID", str(self.proc.processId())],
                             creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        self.proc.kill()
        if self.settings.albedo_enabled and self.settings.albedo_source == "cosmos" \
                and self.settings.cosmos_backend == "wsl":
            from .models.cosmos import kill_wsl_job
            kill_wsl_job(self.settings.wsl_distro)    # and the Linux side, still on the GPU otherwise

    def _read(self):
        self._buf += bytes(self.proc.readAllStandardOutput())
        # progress bars (tqdm) end lines with \r, and can glue our "@@PTA" message to their text
        *lines, self._buf = self._buf.replace(b"\r", b"\n").split(b"\n")
        for raw in lines:
            line = raw.decode("utf-8", "replace")
            k = line.find("@@PTA ")
            if k > 0:
                line = line[k:]
            if line.startswith("@@PTA "):
                try:
                    m = json.loads(line[6:])
                except ValueError:
                    continue
                t = m.get("t")
                if t == "log":
                    self.log.emit(m["m"])
                elif t == "progress":
                    self.progress.emit(m["step"], m["d"], m["n"])
                elif t == "detail":
                    self.detail.emit(m["m"])
                elif t == "finished":
                    self._err = m["err"]
            elif line.strip() and "it/s]" not in line and "s/it]" not in line:
                self.log.emit(line)

    def _done(self, code, status):
        self._read()
        if self.cancelled:
            return self.finished.emit("cancelled")
        if self._err is None:
            self._err = "the process stopped unexpectedly (exit code %s)" % code
        self.finished.emit(self._err)


class SamServer(QtCore.QObject):
    """SAM 3.1 kept loaded in its own process while you click on the People page (pta.sam_server):
    each click re-cuts the current frame right away. Killed when leaving the page or applying."""
    ready = QtCore.Signal()
    preview = QtCore.Signal(int, int, str)          # request id, frame, id image path
    failed = QtCore.Signal(str)
    log = QtCore.Signal(str)

    def __init__(self, settings, parent=None):
        super().__init__(parent)
        self.settings = settings
        self.is_ready = False
        self._id = 0
        self._buf = b""
        self.proc = QtCore.QProcess(self)
        self.proc.setProcessChannelMode(QtCore.QProcess.MergedChannels)
        env = QtCore.QProcessEnvironment.systemEnvironment()
        env.insert("PYTHONIOENCODING", "utf-8")
        env.insert("PYTHONUNBUFFERED", "1")
        self.proc.setProcessEnvironment(env)
        self.proc.setWorkingDirectory(APP_DIR)
        self.proc.readyReadStandardOutput.connect(self._read)
        self.proc.finished.connect(lambda *_: (setattr(self, "is_ready", False)))

    def command(self):
        return PYTHON, ["-u", "-m", "pta.sam_server"]

    def start(self):
        req = os.path.join(tempfile.gettempdir(), "pta_sam_%d.json" % os.getpid())
        with open(req, "w", encoding="utf-8") as f:
            json.dump({"settings": dataclasses.asdict(self.settings)}, f)
        exe, args = self.command()
        self.proc.start(exe, args + [req])

    def request(self, frame, keys):
        self._id += 1
        line = json.dumps({"cmd": "preview", "id": self._id, "frame": int(frame), "keys": keys}) + "\n"
        self.proc.write(line.encode("utf-8"))
        return self._id

    def stop(self):
        if self.proc.state() != QtCore.QProcess.NotRunning:
            self.proc.write(b'{"cmd": "quit"}\n')
            if not self.proc.waitForFinished(1500):
                self.proc.kill()
                self.proc.waitForFinished(1000)
        self.is_ready = False

    def _read(self):
        self._buf += bytes(self.proc.readAllStandardOutput())
        *lines, self._buf = self._buf.replace(b"\r", b"\n").split(b"\n")
        for raw in lines:
            line = raw.decode("utf-8", "replace")
            k = line.find("@@PTA ")
            if k < 0:
                if line.strip() and "it/s]" not in line and "s/it]" not in line:
                    self.log.emit(line)
                continue
            try:
                m = json.loads(line[k + 6:])
            except ValueError:
                continue
            t = m.get("t")
            if t == "ready":
                self.is_ready = True
                self.ready.emit()
            elif t == "preview":
                self.preview.emit(int(m["id"]), int(m["frame"]), m["path"])
            elif t == "error":
                self.failed.emit(m.get("m", "error"))
            elif t == "log":
                self.log.emit(m.get("m", ""))


# ------------------------------------------------------------------------------------- widgets
def _widget(value, choices=None, rng=None, step=None):
    return _guard(_widget_(value, choices, rng, step))


def _widget_(value, choices=None, rng=None, step=None):
    if choices:
        w = QtWidgets.QComboBox()
        for key, label in choices:
            w.addItem(label, key)
        w.setCurrentIndex(max(0, w.findData(value)))
        return w
    if isinstance(value, bool):
        w = QtWidgets.QCheckBox()
        w.setChecked(value)
        return w
    if isinstance(value, int):
        w = QtWidgets.QSpinBox()
        w.setRange(*(rng or (-1000000, 1000000)))
        w.setValue(value)
        return w
    if isinstance(value, float):
        w = QtWidgets.QDoubleSpinBox()
        w.setRange(*(rng or (-1e6, 1e6)))
        w.setDecimals(3)
        w.setSingleStep(step or 0.01)
        w.setValue(value)
        return w
    return QtWidgets.QLineEdit(str(value))


class _NoWheel(QtCore.QObject):
    """the mouse wheel scrolls the page, it does not change a field you are only passing over"""
    def eventFilter(self, obj, ev):
        if ev.type() == QtCore.QEvent.Wheel and not obj.hasFocus():
            ev.ignore()
            return True
        return False


_NO_WHEEL = None


def _guard(w):
    global _NO_WHEEL
    if isinstance(w, (QtWidgets.QAbstractSpinBox, QtWidgets.QComboBox)):
        if _NO_WHEEL is None:
            _NO_WHEEL = _NoWheel()
        w.setFocusPolicy(QtCore.Qt.StrongFocus)
        w.installEventFilter(_NO_WHEEL)
    return w


def _get(w):
    if isinstance(w, QtWidgets.QComboBox):
        return w.currentData()
    if isinstance(w, QtWidgets.QCheckBox):
        return w.isChecked()
    if isinstance(w, (QtWidgets.QSpinBox, QtWidgets.QDoubleSpinBox)):
        return w.value()
    return w.text()


def _set(w, v):
    w.blockSignals(True)
    if isinstance(w, QtWidgets.QComboBox):
        w.setCurrentIndex(max(0, w.findData(v)))
    elif isinstance(w, QtWidgets.QCheckBox):
        w.setChecked(bool(v))
    elif isinstance(w, (QtWidgets.QSpinBox, QtWidgets.QDoubleSpinBox)):
        w.setValue(v)
    else:
        w.setText(str(v))
    w.blockSignals(False)


class Segmented(QtWidgets.QWidget):
    """Apple-style segmented control. changed(key)."""
    changed = QtCore.Signal(str)

    def __init__(self, items):
        super().__init__()
        self.setObjectName("segbar")
        self.setAttribute(QtCore.Qt.WA_StyledBackground, True)
        lay = QtWidgets.QHBoxLayout(self)
        lay.setContentsMargins(3, 3, 3, 3)
        lay.setSpacing(2)
        self.group = QtWidgets.QButtonGroup(self)
        self.buttons = {}
        for key, label in items:
            b = QtWidgets.QToolButton()
            b.setText(label)
            b.setCheckable(True)
            self.group.addButton(b)
            lay.addWidget(b)
            self.buttons[key] = b
            b.clicked.connect(lambda _=False, k=key: self.changed.emit(k))

    def set(self, key):
        for k, b in self.buttons.items():
            b.setChecked(k == key)

    def key(self):
        return next((k for k, b in self.buttons.items() if b.isChecked()), None)

    def show_key(self, key, visible):
        self.buttons[key].setVisible(visible)


# everything else, in the Advanced sheet
SIZES = [(s, s) for s in ("0.4b", "0.8b", "1b", "5b")]
PEOPLE_GROUP = "People finder (SAM 3.1)"
ADVANCED = [
    (PEOPLE_GROUP, [
        ("people_enabled", "Number each person (SAM 3.1)", {}),
        ("people_threshold", "Detection confidence", {"rng": (0.05, 0.95), "step": 0.05}),
        ("people_max", "Most people followed", {"rng": (1, 64)}),
        ("people_detect_interval", "Look for new people every (frames)", {"rng": (1, 50)}),
    ]),
    ("People", [
        ("sapiens_size", "Normal model", {"choices": SIZES}),
        ("seg_size", "Segmentation model", {"choices": SIZES}),
        ("crop_padding", "Crop margin", {"rng": (0, 0.5), "step": 0.01}),
        ("head_fraction", "Head size if no face found", {"rng": (0.1, 0.6), "step": 0.05}),
        ("feather_px", "Crop edge softness", {"rng": (1, 256)}),
        ("crop_chunk", "Keep crops still for (frames)", {"rng": (1, 1000)}),
        ("mask_erode_px", "People mask shrink", {"rng": (0, 64)}),
        ("mask_blur_px", "People mask softness", {"rng": (0, 64)}),
    ]),
    ("Scene", [
        ("moge_resolution_level", "Detail", {"rng": (0, 9)}),
        ("moge_refine_steps", "Refinement", {"rng": (0, 8)}),
        ("vda_max_res", "Depth resolution", {"rng": (256, 4096)}),
    ]),
    ("Flicker removal", [
        ("stab_radius", "Frames on each side", {"rng": (0, 6)}),
        ("stab_sigma", "Strictness", {"rng": (0.005, 0.5), "step": 0.005}),
        ("stab_max_angle", "Anti-ghost angle", {"rng": (0, 90), "step": 0.5}),
    ]),
    ("Alpha: Selection (MatAnyone 2)", [
        ("matanyone_max_side", "Working size (shorter side)", {"rng": (256, 4096)}),
        ("matanyone_warmup", "First frame refinement", {"rng": (0, 50)}),
        ("matanyone_grow", "Close holes in the mask (px)", {"rng": (0, 64)}),
    ]),
    ("Alpha: Foreground (LTX)", [
        ("ltx_max_width", "Max width", {"rng": (256, 1920)}),
        ("ltx_max_height", "Max height", {"rng": (256, 1088)}),
        ("ltx_chunk", "Frames per pass", {"rng": (1, 121)}),
        ("ltx_seed", "Seed", {"rng": (0, 2 ** 31 - 1)}),
    ]),
    ("Albedo", [
        ("cosmos_face_crops", "Cosmos: sharper faces (head crops)", {}),
        ("cosmos_face_size", "Cosmos: head size in its crop", {"rng": (0.2, 0.9), "step": 0.05}),
        ("cosmos_face_edge", "Cosmos: face blend softness", {"rng": (0.002, 0.08), "step": 0.002}),
        ("cosmos_material", "Cosmos: roughness + metallic too", {}),
        ("cosmos_geometry", "Cosmos: its normal + depth too", {}),
        ("cosmos_steps", "Cosmos quality (steps)", {"rng": (1, 50)}),
        ("cosmos_overlap", "Cosmos chunk overlap (frames)", {"rng": (0, 24)}),
        ("cosmos_guidance", "Cosmos guidance", {"rng": (0, 10), "step": 0.1}),
        ("cosmos_backend", "Cosmos runs", {"choices": [("native", "In Windows (Materia port)"),
                                                       ("wsl", "In WSL (official NVIDIA code)")]}),
        ("cosmos_offload", "Cosmos low VRAM mode (WSL)", {}),
        ("albedo_variant", "Marigold model", {"choices": [("appearance", "Albedo, roughness, metal"),
                                                           ("lighting", "Albedo, shading")]}),
    ]),
    ("Files", [
        ("half_float", "Half float", {}),
        ("comparison_layers", "Extra comparison layers", {}),
    ]),
]


class AdvancedDialog(QtWidgets.QDialog):
    def __init__(self, s, parent, people=False, dev=False):
        super().__init__(parent)
        self.setWindowTitle("Segmentation settings" if people else "Advanced settings")
        self.setMinimumWidth(460)
        outer = QtWidgets.QVBoxLayout(self)
        scroll = QtWidgets.QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
        host = QtWidgets.QWidget()
        host.setObjectName("advhost")
        v = QtWidgets.QVBoxLayout(host)
        v.setContentsMargins(18, 6, 18, 6)
        self.w = {}
        for title, rows in ADVANCED:
            if (title == PEOPLE_GROUP) != people:
                continue
            h = QtWidgets.QLabel(title.upper())
            h.setObjectName("h2")
            v.addSpacing(8)
            v.addWidget(h)
            f = QtWidgets.QFormLayout()
            f.setLabelAlignment(QtCore.Qt.AlignLeft)
            f.setHorizontalSpacing(18)
            for field, label, kw in rows:
                if field in DEV_FIELDS and not dev:
                    continue
                w = _widget(getattr(s, field), **kw)
                self.w[field] = w
                lab = QtWidgets.QLabel(label)
                lab.setFixedWidth(230)                 # same column for every group
                w.setMinimumWidth(150)
                f.addRow(lab, w)
            v.addLayout(f)
        self.dev = None
        if not people:
            h = QtWidgets.QLabel("DEVELOPER")
            h.setObjectName("h2")
            v.addSpacing(8)
            v.addWidget(h)
            self.dev = QtWidgets.QCheckBox("Developer mode: show the debug viewers and settings")
            self.dev.setChecked(dev)
            v.addWidget(self.dev)
        v.addStretch(1)
        scroll.setWidget(host)
        outer.addWidget(scroll)
        row = QtWidgets.QHBoxLayout()
        reset = QtWidgets.QPushButton("Defaults")
        ok = QtWidgets.QPushButton("Done")
        ok.setObjectName("primary")
        row.addWidget(reset)
        row.addStretch(1)
        row.addWidget(ok)
        outer.addLayout(row)
        ok.clicked.connect(self.accept)
        reset.clicked.connect(lambda: [_set(w, getattr(Settings(), k)) for k, w in self.w.items()])
        self.resize(520, 330 if people else 680)

    def apply(self, s):
        for k, w in self.w.items():
            setattr(s, k, _get(w))



# ------------------------------------------------------------------------------- models window
class TokenDialog(QtWidgets.QDialog):
    """Hugging Face READ token for the gated models. Never saved by the app (Hugging Face's own
    login file keeps it only if you tick Remember)."""
    def __init__(self, licenses, parent):
        super().__init__(parent)
        self.setWindowTitle("Hugging Face token")
        self.setMinimumWidth(470)
        v = QtWidgets.QVBoxLayout(self)
        links = "<br>".join('<a href="%s" style="color:%s">%s</a>' % (u, ACCENT, n) for n, u in licenses)
        msg = QtWidgets.QLabel(
            "These models are <b>gated</b>: their authors ask you to accept a license first.<br><br>"
            "1. Open each page, log in and accept the license:<br>%s<br><br>"
            "2. Create a token with <b>Read</b> access on "
            '<a href="https://huggingface.co/settings/tokens" style="color:%s">huggingface.co/settings/tokens</a>'
            " and paste it here:" % (links, ACCENT))
        msg.setWordWrap(True)
        msg.setOpenExternalLinks(True)
        msg.setTextFormat(QtCore.Qt.RichText)
        v.addWidget(msg)
        self.edit = QtWidgets.QLineEdit()
        self.edit.setEchoMode(QtWidgets.QLineEdit.Password)
        self.edit.setPlaceholderText("hf_...")
        v.addWidget(self.edit)
        self.remember = QtWidgets.QCheckBox("Remember on this PC (kept by Hugging Face, not by PlateToAOVs)")
        v.addWidget(self.remember)
        row = QtWidgets.QHBoxLayout()
        no = QtWidgets.QPushButton("Cancel")
        ok = QtWidgets.QPushButton("Download")
        ok.setObjectName("primary")
        row.addStretch(1)
        row.addWidget(no)
        row.addWidget(ok)
        v.addLayout(row)
        no.clicked.connect(self.reject)
        ok.clicked.connect(lambda: self.accept() if self.edit.text().strip() else None)


class ModelsDialog(QtWidgets.QDialog):
    """What each feature needs, what is on this PC, and a Download button per feature."""
    changed = QtCore.Signal()

    def __init__(self, s, parent, note=""):
        super().__init__(parent)
        self.s = s
        self.setWindowTitle("Models")
        self.setMinimumWidth(640)
        self.proc = None
        self.token = None
        v = QtWidgets.QVBoxLayout(self)
        v.setContentsMargins(20, 16, 20, 16)
        self.note = QtWidgets.QLabel(note or "Models are downloaded once. The ones you already have (for example "
                                     "in your ComfyUI models) are found by name and not downloaded again.")
        self.note.setWordWrap(True)
        self.note.setObjectName("muted")
        v.addWidget(self.note)
        v.addSpacing(6)
        self.grid = QtWidgets.QGridLayout()
        self.grid.setHorizontalSpacing(14)
        self.grid.setVerticalSpacing(10)
        v.addLayout(self.grid)
        v.addSpacing(8)
        self.cur = QtWidgets.QLabel("")
        self.cur.setObjectName("muted")
        self.bar = QtWidgets.QProgressBar()
        self.bar.setTextVisible(False)
        self.bar.hide()
        v.addWidget(self.cur)
        v.addWidget(self.bar)
        self.logbox = QtWidgets.QPlainTextEdit()
        self.logbox.setReadOnly(True)
        self.logbox.setMaximumBlockCount(2000)
        self.logbox.setFixedHeight(110)
        self.logbox.hide()
        v.addWidget(self.logbox)
        row = QtWidgets.QHBoxLayout()
        self.startup = QtWidgets.QCheckBox("Check at start-up")
        self.b_log = QtWidgets.QPushButton("Details")
        self.b_log.setObjectName("link")
        self.b_log.clicked.connect(lambda: self.logbox.setVisible(not self.logbox.isVisible()))
        self.b_all = QtWidgets.QPushButton("Download what is needed")
        self.b_all.setObjectName("primary")
        self.b_close = QtWidgets.QPushButton("Close")
        row.addWidget(self.startup)
        row.addWidget(self.b_log)
        row.addStretch(1)
        row.addWidget(self.b_close)
        row.addWidget(self.b_all)
        v.addLayout(row)
        self.b_close.clicked.connect(self.close)
        self.b_all.clicked.connect(lambda: self._download([g["key"] for g in self.groups
                                                           if g["needed"] and self.st[g["key"]]]))
        self._fill()

    def _fill(self):
        from . import model_catalog as MC
        self.groups = MC.groups(self.s)
        self.st = MC.status(self.s)
        while self.grid.count():
            w = self.grid.takeAt(0).widget()
            if w:
                w.deleteLater()
        self.buttons = {}
        for r, g in enumerate(self.groups):
            miss = self.st[g["key"]]
            lic = " · ".join('<a href="%s" style="color:%s">%s</a>' % (u, ACCENT, n) for n, u in g["licenses"])
            title = QtWidgets.QLabel("<b>%s</b>  <span style='color:#8e8e93'>%s</span><br>"
                                     "<span style='color:#8e8e93'>%s<br>%s%s</span>"
                                     % (g["title"], g["size"], g["why"], "accept the license: " if g["gated"] else "",
                                        lic))
            title.setTextFormat(QtCore.Qt.RichText)
            title.setWordWrap(True)
            title.setOpenExternalLinks(True)
            if not miss:
                state = "<span style='color:#30d158'>Installed</span>"
            elif g["needed"]:
                state = "<span style='color:#ff9f0a'>Needed</span>"
            else:
                state = "<span style='color:#8e8e93'>Not installed</span>"
            sl = QtWidgets.QLabel(state)
            sl.setTextFormat(QtCore.Qt.RichText)
            b = QtWidgets.QPushButton("Download")
            b.setVisible(bool(miss))
            b.setEnabled(self.proc is None)
            b.clicked.connect(lambda _=False, k=g["key"]: self._download([k]))
            self.buttons[g["key"]] = b
            self.grid.addWidget(title, r, 0)
            self.grid.addWidget(sl, r, 1)
            self.grid.addWidget(b, r, 2)
        self.grid.setColumnStretch(0, 1)
        need = [g for g in self.groups if g["needed"] and self.st[g["key"]]]
        self.b_all.setVisible(bool(need))
        self.b_all.setEnabled(self.proc is None)

    def _download(self, keys):
        if not keys or self.proc is not None:
            return
        gated = [g for g in self.groups if g["key"] in keys and g["gated"]]
        env = QtCore.QProcessEnvironment.systemEnvironment()
        if gated and not self.token:
            try:
                from huggingface_hub import get_token
                have = get_token()
            except Exception:
                have = None
            if not have:
                d = TokenDialog([l for g in gated for l in g["licenses"]], self)
                if not d.exec():
                    return
                self.token = d.edit.text().strip()
                if d.remember.isChecked():
                    try:
                        from huggingface_hub import login
                        login(token=self.token, add_to_git_credential=False)
                    except Exception as e:
                        self.logbox.appendPlainText("could not save the token: %s" % e)
        if self.token:
            env.insert("HF_TOKEN", self.token)
        env.insert("PYTHONIOENCODING", "utf-8")
        self.proc = QtCore.QProcess(self)
        self.proc.setProcessEnvironment(env)
        self.proc.setWorkingDirectory(str(_APP))
        self.proc.setProcessChannelMode(QtCore.QProcess.MergedChannels)
        self.proc.readyReadStandardOutput.connect(self._read)
        self.proc.finished.connect(self._done)
        self.failed = []
        self.cur.setText("Starting...")
        self.bar.setRange(0, 0)
        self.bar.show()
        for b in self.buttons.values():
            b.setEnabled(False)
        self.b_all.setEnabled(False)
        self.b_close.setText("Stop")
        self.b_close.clicked.disconnect()
        self.b_close.clicked.connect(self._stop)
        self.proc.start(PYTHON, ["-u", "-m", "pta.download"] + keys)

    def _read(self):
        import re
        txt = bytes(self.proc.readAllStandardOutput()).decode("utf-8", "replace")
        for part in re.split(r"[\r\n]+", txt):
            part = part.strip()
            if not part:
                continue
            m = re.search(r"(\d+)%\|.*?([\d.]+[kMGT]?B?)/([\d.]+[kMGT]?B?)", part)
            m2 = re.match(r"^(\d+)%$", part)
            if m or m2:
                self.bar.setRange(0, 100)
                self.bar.setValue(int((m or m2).group(1)))
                if m:
                    self.cur.setText(self._file + "   %s / %s" % (m.group(2), m.group(3)))
                continue
            self.logbox.appendPlainText(part)
            if part.startswith("downloading"):
                self._file = os.path.basename(part.split()[-1] if " -> " not in part else part.split(" -> ")[0].split()[-1])
                self.cur.setText(self._file)
                self.bar.setRange(0, 0)
            elif part.startswith("FAILED"):
                self.failed.append(part)
            elif part.startswith("=="):
                self._file = part.strip("= ")
                self.cur.setText("Preparing %s..." % self._file)
    _file = ""

    def _stop(self):
        if self.proc is not None:
            self.proc.kill()

    def _done(self, *_):
        self.proc = None
        self.bar.hide()
        self.b_close.setText("Close")
        self.b_close.clicked.disconnect()
        self.b_close.clicked.connect(self.close)
        if self.failed:
            gated = any(w in " ".join(self.failed).lower() for w in ("gated", "restricted", "403", "401", "token"))
            self.cur.setText("Some downloads failed%s. Details below." %
                             (": accept the license on the model page (logged in), check your token, then retry"
                              if gated else ""))
            self.logbox.show()
        else:
            self.cur.setText("Done.")
        self.token = self.token if not self.failed else None
        self._fill()
        self.changed.emit()

    def closeEvent(self, ev):
        if self.proc is not None:
            self.proc.kill()
            self.proc.waitForFinished(2000)
        super().closeEvent(ev)


# ------------------------------------------------------------------------------------- main window
class Main(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("PlateToAOVs")
        self.setWindowIcon(QtGui.QIcon(os.path.join(ASSETS, "pta_icon.png")))
        self.resize(1560, 920)
        fresh = os.environ.get("PTA_FRESH_TEST") == "1"   # first-install test: settings of its own
        self.qs = QtCore.QSettings("PlateToAOVs", "PlateToAOVs-fresh-test" if fresh else "PlateToAOVs")
        if not fresh and not self.qs.allKeys():   # settings saved before the rename (LiveActionAOVs)
            old_qs = QtCore.QSettings("LiveActionAOVs", "LiveActionAOVs")
            for k in old_qs.allKeys():
                self.qs.setValue(k, old_qs.value(k))
        self.dev = self.qs.value("developer_mode", False, type=bool)
        self.s = self._load_last()
        self.job = None
        self.runner = None
        self.frame = 0
        self.t0 = 0
        self.view = "plate"
        self._build()
        self._refresh_job()

    # ---------------------------------------------------------------- layout
    def _build(self):
        central = QtWidgets.QWidget()
        self.setCentralWidget(central)
        root = QtWidgets.QHBoxLayout(central)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        # ---------- left: viewer
        left = QtWidgets.QWidget()
        lv = QtWidgets.QVBoxLayout(left)
        lv.setContentsMargins(16, 14, 16, 12)
        lv.setSpacing(10)
        self.viewer = Viewer()
        lv.addWidget(self.viewer, 1)

        bar = QtWidgets.QHBoxLayout()
        self.seg = Segmented(MAIN_VIEWS)
        bar.addStretch(1)
        bar.addWidget(self.seg)
        self.more = QtWidgets.QToolButton()
        self.more.setText("More")
        self.more.setPopupMode(QtWidgets.QToolButton.InstantPopup)
        menu = QtWidgets.QMenu(self)
        for v in MORE_VIEWS:
            if v is None:
                menu.addSeparator()
            else:
                menu.addAction(v[1], lambda k=v[0]: self._set_view(k))
        self.more.setMenu(menu)
        bar.addWidget(self.more)
        self.b_over = QtWidgets.QToolButton()
        self.b_over.setText("Over plate")
        self.b_over.setCheckable(True)
        self.b_over.setToolTip("Show the alpha in color over the plate, to check the edges")
        self.b_over.setChecked(self.qs.value("alpha_over", False, type=bool))
        self.b_over.toggled.connect(lambda on: (self.qs.setValue("alpha_over", on), self._show()))
        self.b_over.hide()
        bar.addWidget(self.b_over)
        bar.addStretch(1)
        lv.addLayout(bar)

        tl = QtWidgets.QHBoxLayout()
        self.b_play = QtWidgets.QToolButton()
        self.b_play.setObjectName("play")
        self.b_play.setIcon(_icon("play"))
        self.b_play.setIconSize(QtCore.QSize(22, 22))
        self.b_play.setToolTip("Play (Space)")
        self.b_play.setCheckable(True)
        self.slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.frame_lbl = QtWidgets.QLabel("")
        self.frame_lbl.setObjectName("muted")
        self.frame_lbl.setMinimumWidth(48)
        self.frame_lbl.setAlignment(QtCore.Qt.AlignRight | QtCore.Qt.AlignVCenter)
        tl.addWidget(self.b_play)
        tl.addWidget(self.slider, 1)
        tl.addWidget(self.frame_lbl)
        lv.addLayout(tl)
        root.addWidget(left, 1)

        # ---------- right: two pages, like Sammie-Roto. 1 People (who is who), 2 AOVs (everything else)
        side = QtWidgets.QWidget()
        side.setObjectName("side")
        side.setAttribute(QtCore.Qt.WA_StyledBackground, True)
        side.setFixedWidth(340)
        sv = QtWidgets.QVBoxLayout(side)
        sv.setContentsMargins(22, 20, 22, 18)
        sv.setSpacing(6)
        h1 = QtWidgets.QLabel("PlateToAOVs")
        h1.setObjectName("h1")
        sv.addWidget(h1)
        sv.addSpacing(10)
        self.pages = Segmented([("people", "1   Segmentation"), ("aovs", "2   AOVs")])
        for b in self.pages.buttons.values():
            b.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Fixed)
        sv.addWidget(self.pages)
        sv.addSpacing(12)
        self.stack = QtWidgets.QStackedWidget()
        sv.addWidget(self.stack, 1)

        # ===== page 1: plate + people
        p1 = QtWidgets.QWidget()
        v1 = QtWidgets.QVBoxLayout(p1)
        v1.setContentsMargins(0, 0, 0, 0)
        v1.setSpacing(6)
        v1.addWidget(self._h2("PLATE"))
        self.plate_lbl = QtWidgets.QLabel("No plate")
        self.plate_lbl.setObjectName("plate")
        self.plate_info = QtWidgets.QLabel("")
        self.plate_info.setObjectName("muted")
        self.plate_info.setWordWrap(True)
        b_choose = QtWidgets.QPushButton("Choose plate...")
        v1.addWidget(self.plate_lbl)
        v1.addWidget(self.plate_info)
        v1.addWidget(b_choose)
        v1.addSpacing(6)
        self.cs = _widget(self.s.colorspace, [("srgb", "sRGB (PNG, JPG)"), ("linear", "Linear (EXR)"),
                                              ("acescg", "ACEScg (EXR)")])
        form = QtWidgets.QFormLayout()
        form.setHorizontalSpacing(12)
        form.addRow(self._muted("Color"), self.cs)
        rng = QtWidgets.QHBoxLayout()
        self.first, self.last = _widget(self.s.first, rng=(0, 10 ** 7)), _widget(self.s.last, rng=(0, 10 ** 7))
        self._full = None                         # first / last frame of the whole plate
        for w in (self.first, self.last):
            w.setKeyboardTracking(False)          # typing 1050 does not load 1, 10, 105 first
            w.setToolTip("Frame range to work on (frame numbers of the plate). The whole plate by default.")
        rng.addWidget(self.first, 1)
        rng.addSpacing(4)
        rng.addWidget(self._muted("to"))
        rng.addSpacing(4)
        rng.addWidget(self.last, 1)
        form.addRow(self._muted("Frames"), rng)
        v1.addLayout(form)

        self.people_box = QtWidgets.QWidget()
        pv = QtWidgets.QVBoxLayout(self.people_box)
        pv.setContentsMargins(0, 14, 0, 0)
        pv.setSpacing(6)
        pv.addWidget(self._h2("PEOPLE AND OBJECTS"))
        lookrow = QtWidgets.QHBoxLayout()
        self.look = QtWidgets.QLineEdit(self.s.people_prompt)
        self.look.setPlaceholderText("person, car, house...")
        self.look.setToolTip("What Find everyone looks for, in English, separated by commas.\n"
                             "Anything that is not a person is numbered as an object.")
        lookrow.addWidget(self._muted("Find"))
        lookrow.addWidget(self.look, 1)
        pv.addLayout(lookrow)
        self.people_msg = self._muted("")
        self.people_msg.setWordWrap(True)
        pv.addWidget(self.people_msg)
        self.people_rows_host = QtWidgets.QWidget()
        self.people_rows = QtWidgets.QVBoxLayout(self.people_rows_host)
        self.people_rows.setContentsMargins(0, 0, 0, 0)
        self.people_rows.setSpacing(4)
        rows_scroll = self.rows_scroll = QtWidgets.QScrollArea()
        rows_scroll.setWidgetResizable(True)
        rows_scroll.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
        rows_scroll.setWidget(self.people_rows_host)
        rows_scroll.setMaximumHeight(204)
        self.people_rows_host.setObjectName("advhost")
        pv.addWidget(rows_scroll)
        self.fix_box = QtWidgets.QWidget()
        fv = QtWidgets.QVBoxLayout(self.fix_box)
        fv.setContentsMargins(0, 8, 0, 0)
        fv.setSpacing(6)
        fixrow = QtWidgets.QHBoxLayout()
        self.b_fix = QtWidgets.QPushButton("Select with clicks")
        self.b_fix.setCheckable(True)
        self.fix_target = _guard(QtWidgets.QComboBox())
        fixrow.addWidget(self.b_fix)
        fixrow.addWidget(self.fix_target, 1)
        fv.addLayout(fixrow)
        applyrow = QtWidgets.QHBoxLayout()
        self.b_apply = QtWidgets.QPushButton("Propagate")
        self.b_apply.setToolTip("Follow everyone through the shot with all your keyframes (SAM 2.1)")
        self.b_clear = QtWidgets.QPushButton("Undo click")
        self.b_clearf = QtWidgets.QPushButton("Clear frame")
        self.b_clearf.setToolTip("Remove the clicks of the selected person or object on this frame")
        applyrow.addWidget(self.b_apply)
        applyrow.addWidget(self.b_clear)
        applyrow.addWidget(self.b_clearf)
        fv.addLayout(applyrow)
        self.fix_hint = self._muted("")           # under the buttons: they stay together when it shows
        self.fix_hint.setWordWrap(True)
        fv.addWidget(self.fix_hint)
        pv.addWidget(self.fix_box)
        v1.addWidget(self.people_box)
        b_padv = QtWidgets.QPushButton("Settings...")
        b_padv.setObjectName("link")
        b_padv.setToolTip("Find settings: detection confidence, how many at most...")
        self.b_clearall = QtWidgets.QPushButton("Clear all")
        self.b_clearall.setObjectName("link")
        self.b_clearall.setToolTip("Start again: removes everyone, your clicks, names and objects")
        self.b_clearall.clicked.connect(self._clear_all)
        linkrow = QtWidgets.QHBoxLayout()
        linkrow.addWidget(b_padv)
        linkrow.addStretch(1)
        linkrow.addWidget(self.b_clearall)
        v1.addLayout(linkrow)
        v1.addStretch(1)
        self.stack.addWidget(p1)
        self.pending = {}                          # (frame, person) -> keyframe not propagated yet (None = cleared)
        self.history = []                          # (frame, person) per click, for Undo
        self.previews = {}                         # frame -> id image path (live clicks)
        self._want = {}                            # frame -> request id we are waiting for
        self.sam = None
        self._people_sig = None

        # ===== page 2: passes + output
        p2 = QtWidgets.QWidget()
        v2 = QtWidgets.QVBoxLayout(p2)
        v2.setContentsMargins(0, 0, 0, 0)
        v2.setSpacing(6)
        v2.addWidget(self._h2("PLATE"))
        self.plate_sum = QtWidgets.QLabel("")
        self.plate_sum.setObjectName("plate")
        self.people_sum = self._muted("")
        self.people_sum.setWordWrap(True)
        v2.addWidget(self.plate_sum)
        v2.addWidget(self.people_sum)
        v2.addSpacing(16)
        v2.addWidget(self._h2("PASSES"))
        self.c_normals = QtWidgets.QCheckBox("Normals")
        self.c_normals.toggled.connect(lambda on: self.c_head.setEnabled(on))
        self.c_depth = QtWidgets.QCheckBox("Depth and position")
        self.c_alpha = QtWidgets.QCheckBox("Alpha")
        self.c_alpha.toggled.connect(lambda on: self.alpha_q.setEnabled(on))
        self.alpha_q = _widget(self.s.alpha_source, [("matanyone", "Selection (MatAnyone 2)"),
                                                     ("ltx", "Foreground (LTX)"), ("matting", "Fast (people)"),
                                                     ("sapiens", "Instant (hard edge)")])
        self.alpha_q.setToolTip("Selection: the alpha of what you selected on page 1, one layer per number.\n"
                                "Foreground: LTX decides what the foreground is, on the whole frame.")
        self.c_albedo = QtWidgets.QCheckBox("Albedo")
        self.albedo_q = _widget(self.s.albedo_source, [("cosmos", "Best (Cosmos)"), ("marigold", "Fast (Marigold)")])
        self.c_albedo.toggled.connect(lambda on: self.albedo_q.setEnabled(on))
        self.c_head = QtWidgets.QCheckBox("Extra face detail")
        self.fov = _widget(self.s.fov_x, rng=(0, 170), step=1.0)
        self.fov.setDecimals(1)
        self.fov.setSuffix(" °")
        self.fov.setSpecialValueText("auto")
        v2.addWidget(self.c_normals)
        hr = QtWidgets.QHBoxLayout()
        hr.setContentsMargins(25, 0, 0, 0)
        hr.addWidget(self.c_head)
        v2.addLayout(hr)
        v2.addWidget(self.c_depth)
        ar = QtWidgets.QHBoxLayout()
        ar.addWidget(self.c_alpha)
        ar.addWidget(self.alpha_q, 1)
        v2.addLayout(ar)
        alr = QtWidgets.QHBoxLayout()
        alr.addWidget(self.c_albedo)
        alr.addWidget(self.albedo_q, 1)
        v2.addLayout(alr)
        fr = QtWidgets.QFormLayout()
        fr.addRow(self._muted("Camera FOV"), self.fov)
        v2.addLayout(fr)
        b_adv = QtWidgets.QPushButton("Advanced settings...")
        b_adv.setObjectName("link")
        v2.addWidget(b_adv)
        v2.addStretch(1)
        v2.addWidget(self._h2("OUTPUT"))
        self.out_lbl = QtWidgets.QPushButton("")
        self.out_lbl.setObjectName("link")
        self.out_lbl.setToolTip("Click to choose another folder")
        v2.addWidget(self.out_lbl)
        self.stack.addWidget(p2)

        # ===== bottom, both pages
        sv.addSpacing(10)
        self.status = QtWidgets.QLabel("")
        self.status.setWordWrap(True)
        self.bar = QtWidgets.QProgressBar()
        self.bar.setTextVisible(False)
        self.bar.hide()
        sv.addWidget(self.status)
        sv.addWidget(self.bar)
        self.b_run = QtWidgets.QPushButton("Generate")
        self.b_run.setObjectName("primary")
        self.b_cancel = QtWidgets.QPushButton("Cancel")
        self.b_cancel.hide()
        sv.addWidget(self.b_run)
        sv.addWidget(self.b_cancel)
        low = QtWidgets.QHBoxLayout()
        self.b_open = QtWidgets.QPushButton("Show in Explorer")
        self.b_open.setObjectName("link")
        self.b_log = QtWidgets.QPushButton("Log")
        self.b_log.setObjectName("link")
        self.b_models = QtWidgets.QPushButton("Models")
        self.b_models.setObjectName("link")
        low.addWidget(self.b_open)
        low.addStretch(1)
        low.addWidget(self.b_models)
        low.addSpacing(12)
        low.addWidget(self.b_log)
        sv.addLayout(low)
        root.addWidget(side)

        self.logwin = QtWidgets.QPlainTextEdit()
        self.logwin.setReadOnly(True)
        self.logwin.setMaximumBlockCount(5000)
        self.logwin.setWindowTitle("PlateToAOVs log")
        self.logwin.resize(760, 460)

        # ---------- signals
        b_choose.clicked.connect(self._pick_plate)
        self.look.editingFinished.connect(self._refresh_job)
        self.out_lbl.clicked.connect(self._pick_out)
        for w in (self.cs, self.first, self.last, self.alpha_q, self.albedo_q, self.fov):
            for sig in ("currentIndexChanged", "valueChanged"):
                if hasattr(w, sig):
                    getattr(w, sig).connect(lambda *_: self._refresh_job())
        for c in (self.c_normals, self.c_alpha, self.c_depth, self.c_albedo, self.c_head):
            c.toggled.connect(lambda *_: self._refresh_job())
        b_adv.clicked.connect(self._advanced)
        b_padv.clicked.connect(lambda: self._advanced(people=True))
        self.pages.changed.connect(self._set_page)
        self.b_fix.toggled.connect(self._fix_mode)
        self.b_apply.clicked.connect(self._apply_fixes)
        self.b_clear.clicked.connect(self._clear_clicks)
        self.b_clearf.clicked.connect(self._clear_frame)
        self.viewer.clicked.connect(self._view_click)
        self.seg.changed.connect(self._set_view)
        self.slider.valueChanged.connect(self._set_frame)
        self.b_play.toggled.connect(self._play)
        self.timer = QtCore.QTimer(self)
        self.timer.timeout.connect(self._tick)
        self.b_run.clicked.connect(lambda *_: self._primary())
        self.b_cancel.clicked.connect(self._cancel)
        self.b_open.clicked.connect(self._open_out)
        self.b_log.clicked.connect(lambda: (self.logwin.show(), self.logwin.raise_()))
        self.b_models.clicked.connect(lambda: self._models())
        QtCore.QTimer.singleShot(400, self._models_at_start)
        for i in range(len(MAIN_VIEWS)):                  # 1, 2, 3... = the visible view buttons, in order
            QtGui.QShortcut(QtGui.QKeySequence(str(i + 1)), self, activated=lambda i=i: self._view_key(i))
        QtGui.QShortcut(QtGui.QKeySequence("Left"), self, activated=lambda: self.slider.setValue(self.slider.value() - 1))
        QtGui.QShortcut(QtGui.QKeySequence("Right"), self, activated=lambda: self.slider.setValue(self.slider.value() + 1))
        QtGui.QShortcut(QtGui.QKeySequence("F"), self, activated=self.viewer.fit)
        QtGui.QShortcut(QtGui.QKeySequence("Space"), self, activated=self.b_play.toggle)
        self._apply_quick(self.s)
        self.page = "people"
        self._set_page("people")

    def _h2(self, t):
        l = QtWidgets.QLabel(t)
        l.setObjectName("h2")
        return l

    def _muted(self, t):
        l = QtWidgets.QLabel(t)
        l.setObjectName("muted")
        return l

    # ---------------------------------------------------------------- settings
    def _load_last(self):
        try:
            s = Settings.from_dict(json.loads(self.qs.value("last_settings", "") or "{}"))
        except Exception:
            s = Settings()
        if not self.qs.value("faces_off_v1", False):   # Cosmos face crops are off by default now
            s.cosmos_face_crops = False
            self.qs.setValue("faces_off_v1", True)
        if not self.qs.value("ma2_720_v1", False):     # MatAnyone 2 leaks on neighbours at 1080: 720 now
            s.matanyone_max_side = Settings().matanyone_max_side
            self.qs.setValue("ma2_720_v1", True)
        return s

    def _apply_quick(self, s):
        _set(self.cs, s.colorspace)
        _set(self.first, s.first)
        _set(self.last, s.last)
        _set(self.alpha_q, s.alpha_source)
        self.look.setText(s.people_prompt)
        _set(self.fov, s.fov_x)
        _set(self.c_depth, s.depth_enabled)
        _set(self.c_albedo, s.albedo_enabled)
        _set(self.c_head, s.head_pass)
        _set(self.c_normals, s.normals_enabled)
        _set(self.c_alpha, s.alpha_enabled)
        _set(self.albedo_q, s.albedo_source)
        self.albedo_q.setEnabled(s.albedo_enabled)
        self.c_head.setEnabled(s.normals_enabled)
        self.alpha_q.setEnabled(s.alpha_enabled)

    def _collect(self):
        s = dataclasses.replace(self.s)
        s.colorspace = _get(self.cs)
        s.first, s.last = _get(self.first), _get(self.last)
        if self._full:                             # the whole plate is stored as 0 / 0 (= start / end)
            if s.first and s.last and s.first > s.last:
                s.first, s.last = s.last, s.first
            s.first = 0 if s.first <= self._full[0] else s.first
            s.last = 0 if s.last >= self._full[1] else s.last
        else:
            s.first, s.last = self.s.first, self.s.last
        s.alpha_source = _get(self.alpha_q)
        s.people_prompt = ", ".join(p.strip() for p in self.look.text().split(",") if p.strip()) or "person"
        s.fov_x = _get(self.fov)
        s.depth_enabled = self.c_depth.isChecked()
        s.albedo_enabled = self.c_albedo.isChecked()
        s.head_pass = self.c_head.isChecked()
        s.normals_enabled = self.c_normals.isChecked()
        s.alpha_enabled = self.c_alpha.isChecked()
        s.albedo_source = _get(self.albedo_q)
        s.people_review = False                    # the People page is where the numbers get checked
        return s

    def _advanced(self, people=False):
        d = AdvancedDialog(self.s, self, people=people, dev=self.dev)
        if d.exec():
            d.apply(self.s)
            if d.dev is not None and d.dev.isChecked() != self.dev:
                self.dev = d.dev.isChecked()
                self.qs.setValue("developer_mode", self.dev)
                self._set_page(self.page)
            self._refresh_job()

    def _pass_view(self, key):
        """AOVs page: a view button shows when its pass is on (debug ones in developer mode only)."""
        on = {"albedo": self.s.albedo_enabled, "depth": self.s.depth_enabled, "alpha": self.s.alpha_enabled,
              "final": self.s.normals_enabled}.get(key, True)
        return on and (self.dev or key not in DEV_MAIN_VIEWS)

    def _view_visible(self, k):
        if k in self.seg.buttons:
            return not self.seg.buttons[k].isHidden()
        return self.dev and self.page != "people"   # "More" menu

    def _refresh_job(self):
        self.s = self._collect()
        try:
            self.qs.setValue("last_settings", json.dumps(dataclasses.asdict(self.s)))
        except Exception:
            pass
        self.job = None
        if getattr(self, "page", "people") != "people":
            for key in self.seg.buttons:
                self.seg.show_key(key, self._pass_view(key))
            if not self._view_visible(self.view):
                self._set_view(next((k for k in ("final", "alpha", "depth", "albedo") if self._pass_view(k)), "plate"))
        if not self.s.input_pattern:
            self.plate_lbl.setText("No plate")
            self.plate_info.setText("Choose one frame of an image sequence.")
            self.out_lbl.setText("")
            self.viewer.show_image(None, "Choose a plate to begin")
            return self._update_status()
        try:
            self.job = P.Job(self.s, log=lambda m: None)
        except Exception as e:
            self.plate_lbl.setText("Plate not found")
            self.plate_info.setText(str(e))
            self.viewer.show_image(None, "")
            return self._update_status()
        self._offer_previous()
        j = self.job
        self.plate_lbl.setText(j.name)
        a, b = self._full = j.full_range
        for w, v in ((self.first, self.s.first or a), (self.last, self.s.last or b)):
            w.blockSignals(True)
            w.setRange(a, b)
            w.setValue(v)
            w.blockSignals(False)
        whole = b - a + 1
        self.plate_info.setText("%d frames  ·  %s–%s%s" % (j.n, j.frame_label(0), j.frame_label(j.n - 1),
                                                           "" if j.n >= whole else "  (of %d)" % whole))
        self.out_lbl.setText(_short(j.out))
        self.slider.blockSignals(True)
        self.slider.setRange(0, j.n - 1)
        self.slider.setValue(min(self.frame, j.n - 1))
        self.slider.blockSignals(False)
        self._update_status()
        self._show()
        self._preload()

    def _offer_previous(self):
        jf = os.path.join(self.job.out, "pta_job.json")
        if not os.path.isfile(jf):                # a folder made before the rename
            jf = os.path.join(self.job.out, "laa_job.json")
        if not os.path.isfile(jf) or getattr(self, "_offered", None) == jf:
            return
        self._offered = jf
        try:
            with open(jf) as f:
                old = Settings.from_dict(json.load(f))
        except Exception:
            return
        if old != self.s and QtWidgets.QMessageBox.question(
                self, "PlateToAOVs", "This plate was already processed.\nUse the same settings?") == QtWidgets.QMessageBox.Yes:
            self.s = old
            self._apply_quick(old)
            self.s = self._collect()
            self.job = P.Job(self.s, log=lambda m: None)

    def _update_status(self):
        if self.runner is not None:
            return
        if self.job is None:
            self.status.setText("<span style='color:#8e8e93'>Choose a plate to begin.</span>" if self.page == "people" else "")
            self.b_run.setText(self._find_label() if self.page == "people" else "Generate")
            self.b_run.setEnabled(False)
            self.people_box.hide()
            return
        self._refresh_people()
        if self.page == "people":
            return self._people_status()
        self.plate_sum.setText("%s  ·  %d frames" % (self.job.name, self.job.n))
        if self.job.people_on() and self.job.is_fresh("people"):
            e = self.job.people_edits()
            ids = self.job._read_done("people").get("ids", [])
            kept = [k for k in ids if k not in set(int(x) for x in e.get("removed", [])) and str(k) not in e.get("same", {})]
            self.people_sum.setText("%s selected: %s" % (self._count_text(kept, e).capitalize(),
                                                         ", ".join(self._pname(k, e) for k in kept))
                                    if kept else "Nothing selected")
        elif self.job.people_on():
            self.people_sum.setText("Nothing selected yet: Generate will find people automatically.")
        else:
            self.people_sum.setText("Segmentation off")
        steps = self.job.active_steps()
        if "write" not in steps:
            self.status.setText("<span style='color:#8e8e93'>Turn on at least one pass.</span>")
            self.b_run.setText("Generate")
            self.b_run.setEnabled(False)
            return
        self.b_run.setEnabled(True)
        todo, stale = [], set()
        for name in steps:
            try:
                fresh = self.job.is_fresh(name) and not (set(self.job.deps(name)[1]) & stale)
            except Exception:
                fresh = False
            if not fresh:
                stale.add(name)
                todo.append(name)
        if not todo:
            self.status.setText("<span style='color:#30d158'>\u2713</span>  All passes are generated")
            self.b_run.setText("Up to date")
            self.b_run.setEnabled(False)
        else:
            done = len(steps) - len(todo)
            self.status.setText("<span style='color:#8e8e93'>Ready. %s</span>" %
                                ("Nothing computed yet." if done == 0 else "%d of %d steps already done." % (done, len(steps))))
            self.b_run.setText("Generate")

    def _people_status(self):
        """page 1: Find people -> check / fix -> Next"""
        self.b_run.setEnabled(True)
        if not self.s.people_enabled:
            self.status.setText("<span style='color:#8e8e93'>People finder is off (Segmentation > Settings): "
                                "people are told apart by touching shapes.</span>")
            self.b_run.setText("Next: AOVs  →")
        elif not self._plate_ready():
            self.status.setText("<span style='color:#8e8e93'>Loading the plate...</span>")
            self.b_run.setText(self._find_label())
            self.b_run.setEnabled(False)
        elif not self._people_ready():
            self.status.setText("<span style='color:#8e8e93'>Click on what you want to select, or let SAM 3.1 "
                                "find it (Find).</span>")
            self.b_run.setText(self._find_label())
        else:
            self.status.setText("<span style='color:#30d158'>✓</span>  Check the numbers on a few frames, "
                                "fix them if needed, then go to AOVs.")
            self.b_run.setText("Next: AOVs  →")

    def _primary(self):
        if self.page == "people":
            if self.job is not None and self.s.people_enabled and not self._people_ready():
                e = self.job.people_edits()
                if e.get("keys") and QtWidgets.QMessageBox.question(
                        self, "PlateToAOVs", "Finding everyone automatically starts the selection again "
                        "(your clicks are dropped). Continue?") != QtWidgets.QMessageBox.Yes:
                    return
                e["keys"], e["auto"] = [], True   # SAM 3.1 numbers people its own way: start clean
                e["names"], e["kinds"] = {}, {}
                e["removed"], e["same"] = [], {}  # they were about the old numbers
                self.job.save_people_edits(e)
                self.pending, self.history = {}, []
                self.previews.clear()
                return self._run(until="people")
            if self.pending and QtWidgets.QMessageBox.question(
                    self, "PlateToAOVs", "You have clicks not applied yet.\nApply them first?") == QtWidgets.QMessageBox.Yes:
                return self._apply_fixes()
            self.pending = {}
            if self._people_ready():
                e = self.job.people_edits()
                e["validated"] = True
                self.job.save_people_edits(e)
            return self._set_page("aovs")
        self._run()

    def _set_page(self, k):
        self.page = k
        self.pages.set(k)
        self.stack.setCurrentIndex(0 if k == "people" else 1)
        if k != "people":
            self.b_fix.setChecked(False)
            self._stop_sam()
        for key in self.seg.buttons:
            self.seg.show_key(key, key in ("plate", "people") if k == "people" else self._pass_view(key))
        self.more.setVisible(k != "people" and self.dev)
        if k == "people":
            self._set_view("people" if self._plate_ready() else "plate")
        elif self.view in ("plate", "people") or not self._view_visible(self.view):
            self._set_view("final" if self.job and os.path.isfile(self.job.ppath("final", 0)) else self.view)
        self._update_status()

    # ---------------------------------------------------------------- viewer
    def _set_view(self, k):
        self.view = k
        self.seg.set(k)
        self.more.setChecked(False)
        self.more.setText("More" if k in dict(MAIN_VIEWS) else VIEW_NAMES[k])
        self._show()

    def _view_key(self, i):
        keys = [k for k, _ in MAIN_VIEWS if not self.seg.buttons[k].isHidden()]
        if i < len(keys):
            self._set_view(keys[i])

    def _set_frame(self, v):
        self.frame = v
        self._show()

    def _show(self):
        if self.job is None:
            return
        i = min(self.frame, self.job.n - 1)
        self.frame_lbl.setText(self.job.frame_label(i))
        self.b_over.setVisible(self.view == "alpha")
        if self.view == "alpha" and self.b_over.isChecked():
            img = self._alpha_over(i)
            if img is not None:
                return self.viewer.show_array(img)
        if self.view == "people":
            img = self._people_image(i)
            if img is not None:
                return self.viewer.show_array(img)
            if not os.path.isfile(self.job.ppath("people", i)):
                return self.viewer.show_image(self.job.ppath("plate", i), "Loading the plate...")
        self.viewer.show_image(self.job.ppath(self.view, i), "Not generated yet")

    def _alpha_over(self, i):
        """plate with the alpha in red, half transparent (like a roto overlay)"""
        import cv2
        pa, pp = self.job.ppath("alpha", i), self.job.ppath("plate", i)
        if not (os.path.isfile(pa) and os.path.isfile(pp)):
            return None
        rgb = cv2.imread(pp)[..., ::-1].astype(np.float32)
        a = cv2.imread(pa, cv2.IMREAD_GRAYSCALE)
        if a.shape != rgb.shape[:2]:
            a = cv2.resize(a, (rgb.shape[1], rgb.shape[0]), interpolation=cv2.INTER_LINEAR)
        a = (a.astype(np.float32) / 255.0)[..., None] * 0.55
        tint = np.array([255, 59, 48], np.float32)            # system red
        return np.ascontiguousarray(np.clip(rgb * (1 - a) + tint * a, 0, 255).astype(np.uint8))

    def _play(self, on):
        self.timer.start(42) if on else self.timer.stop()
        self.b_play.setIcon(_icon("pause" if on else "play"))

    def _tick(self):
        if self.job:
            self.slider.setValue((self.slider.value() + 1) % self.job.n)

    # ---------------------------------------------------------------- run
    # ---------------------------------------------------------------- models
    def _missing_needed(self, until=None):
        """features needed by this run whose model files are not on this PC"""
        from . import model_catalog as MC
        if until == "plate":
            return []
        keys = ["people"] if until == "people" else None
        st = MC.status(self.s)
        return [g["key"] for g in MC.groups(self.s) if g["needed"] and st[g["key"]]
                and (keys is None or g["key"] in keys)]

    def _models(self, note=""):
        d = getattr(self, "_models_win", None)
        if d is None or not d.isVisible():
            d = self._models_win = ModelsDialog(self.s, self, note)
            d.startup.setChecked(self.qs.value("models_at_start", True, type=bool))
            d.startup.toggled.connect(lambda on: self.qs.setValue("models_at_start", on))
            d.changed.connect(lambda: (P.config._FOUND.clear(), self._refresh_job()))
        elif note:
            d.note.setText(note)
        d.show()
        d.raise_()
        return d

    def _models_at_start(self):
        if not self.qs.value("models_at_start", True, type=bool):
            return
        try:
            miss = self._missing_needed()
        except Exception:
            return
        if miss:
            self._models("Some models needed by the passes you have turned on are not on this PC yet. "
                         "Download them here (once), or turn those passes off.")

    def _run(self, until=None):
        if self.runner is not None:               # one job at a time (the plate may still be loading)
            return
        self._stop_sam()
        self._refresh_job()
        if self.job is None:
            return
        miss = self._missing_needed(until)
        if miss:
            self._models("To run this, download the models marked Needed first.")
            return
        self.b_fix.setChecked(False)
        self.logwin.appendPlainText("---- %s" % time.strftime("%H:%M:%S"))
        self.steps = self.job.active_steps()
        self.cur_step, self.cur_detail = None, ""
        self._last_until = until
        self.runner = Runner(self.s, parent=self, until=until)
        self.runner.log.connect(self._log)
        self.runner.progress.connect(self._progress)
        self.runner.detail.connect(self._detail)
        self.runner.finished.connect(self._finished)
        self.b_run.hide()
        self.b_cancel.show()
        self.bar.setRange(0, 0)
        self.bar.show()
        self.status.setText("Starting...")
        self.t0 = time.time()
        self.runner.start()

    def _cancel(self):
        if self.runner:
            self.runner.cancel()          # kills the worker process: stops right away

    def _status_line(self, d=None, t=None):
        step = self.cur_step
        if step is None:
            return
        k = self.steps.index(step) + 1 if step in self.steps else 0
        what = STEP_TEXT.get(step, step)
        extra = self.cur_detail
        if not extra:
            extra = "%d / %d" % (d, t) if d else "loading..."
        self.status.setText("%s  <span style='color:#8e8e93'>%s  \u00b7  step %d of %d</span>"
                            % (what, extra, k, len(self.steps)))

    def _log(self, m):
        self.logwin.appendPlainText(m)
        if m.startswith("[") and "] running" in m:
            name = m[1:m.index("]")]
            self.cur_step, self.cur_detail = name, ""
            self.bar.setRange(0, 0)               # busy until the first frame is done
            self._status_line()
            v = STEP_VIEW.get(name)                # the viewer follows the step being computed
            if self.page == "people" and v not in ("plate", "people"):
                v = None                           # People page: depth etc. load in the background
            if v and not self._view_visible(v):
                v = None                           # a debug view, hidden outside developer mode
            if v and v != self.view and not self.b_play.isChecked():
                self._set_view(v)
                self.slider.setValue(0)

    def _detail(self, text):
        self.cur_detail = text
        self._status_line()

    def _progress(self, step, d, t):
        self.cur_step = step
        self.bar.setRange(0, max(t, 1))
        self.bar.setValue(d)
        if step != "alpha":                       # LTX writes its own detail (step 3 of 8...)
            self.cur_detail = ""
        self._status_line(d, t)
        if self.job and d > 0 and STEP_VIEW.get(step) == self.view and not self.b_play.isChecked():
            self.slider.setValue(min(d - 1, self.job.n - 1))

    def _finished(self, err):
        self.runner = None
        self.b_cancel.hide()
        self.b_run.show()
        self.bar.hide()
        self._refresh_job()
        if err == "" and getattr(self, "_last_until", None) == "plate":
            self.status.setText("")
            self._update_status()
            if self.page == "people":
                self._set_view("people")
            self._show()
            return
        if err == "" and getattr(self, "_last_until", None) == "people":
            err = "review"                        # fixes applied: back to checking the numbers
        if err == "review":
            self._set_view("people")
            self._update_status()
            if self.pending:
                self.pending = {}
            self._show()
            return
        if err == "cancelled":
            self.status.setText("Cancelled.")
            self.logwin.appendPlainText("cancelled")
        elif err:
            self.status.setText("<span style='color:#ff453a'>Something went wrong.</span> "
                                "<span style='color:#8e8e93'>%s</span>" % err[:200])
            self.logwin.show()
        else:
            m, s = divmod(int(time.time() - self.t0), 60)
            self.status.setText("<span style='color:#30d158'>\u2713</span>  Done in %s" % ("%dm %02ds" % (m, s) if m else "%ds" % s))
            if self.s.normals_enabled:
                self._set_view("final")

    # ---------------------------------------------------------------- people review
    def _people_ready(self):
        try:
            return self.job is not None and self.s.people_enabled and self.job.is_fresh("people")
        except Exception:
            return False

    def _plate_ready(self):
        try:
            return self.job is not None and self.job.is_fresh("plate")
        except Exception:
            return False

    def _preload(self):
        """as soon as a plate is chosen: frames (and depth, for the people's depth filter) into the
        cache, so you can click on people right away."""
        if self.job is None or self.runner is not None or not self.s.input_pattern:
            return
        until = "plate"
        try:
            todo = not self.job.is_fresh("plate")
        except Exception:
            todo = False
        key = (self.s.input_pattern, self.s.first, self.s.last, self.s.colorspace, until)
        if todo and getattr(self, "_preloaded", None) != key:
            self._preloaded = key
            QtCore.QTimer.singleShot(0, lambda: self._run(until=until))

    def _refresh_people(self):
        """rows: colour, number, keep, same as. Rebuilt only when the numbers or decisions change."""
        if not self._plate_ready() or not self.s.people_enabled:
            self.people_box.hide()
            self.b_fix.setChecked(False)
            return
        ready = self._people_ready()
        done = self.job._read_done("people") if ready else {}
        ids = done.get("ids", [])
        frames = done.get("frames", {})
        e = self.job.people_edits()
        extra = sorted({int(k["target"]) for k in self._keys()} - set(ids))     # clicked, not propagated yet
        sig = (tuple(ids), tuple(extra), json.dumps(e, sort_keys=True), done.get("run"))
        self.people_box.show()
        removed = set(int(k) for k in e.get("removed", []))
        same = {int(k): int(v) for k, v in e.get("same", {}).items()}
        kept = [k for k in ids if k not in removed and k not in same]
        if not ready:
            self.people_msg.setText("Nobody selected yet: press Find below, or Select with clicks (New person / "
                                    "New object) and click on them.")
        else:
            self.people_msg.setText("%s. Untick one to leave it out, or say two numbers are the same."
                                    % self._count_text(kept, e).capitalize() if ids else
                                    "Nothing yet. Select with clicks, New person / New object, click on them.")
        self.rows_scroll.setVisible(bool(ids or extra))
        if sig == self._people_sig:
            return
        self._people_sig = sig
        while self.people_rows.count():
            it = self.people_rows.takeAt(0)
            if it.widget():
                it.widget().deleteLater()
        from .stages import people as PP
        for k in list(ids) + extra:
            new = k in extra
            obj = self._kind(k, e)[0] == "object"
            row = QtWidgets.QWidget()
            h = QtWidgets.QHBoxLayout(row)
            h.setContentsMargins(0, 0, 4, 0)
            sw = QtWidgets.QLabel()
            sw.setFixedSize(12, 12)                    # round = person, square = object
            sw.setStyleSheet("background: rgb(%d,%d,%d); border-radius: %dpx;" % (PP.color(k) + ((2,) if obj else (6,))))
            sw.setToolTip("object" if obj else "person")
            c = QtWidgets.QCheckBox("%d" % k)
            c.setChecked(k not in removed)
            c.setToolTip("keep %d (on screen in %s of %d frames)" % (k, frames.get(str(k), "?"), self.job.n))
            nm = QtWidgets.QLineEdit(e.get("names", {}).get(str(k), ""))
            nm.setPlaceholderText(self._pname(k, dict(e, names={})))
            nm.setToolTip("Name of %d (goes in the EXR header with its people.id value)" % k)
            nm.setEnabled(k not in removed)
            nm.editingFinished.connect(lambda k=k, nm=nm: self._rename(k, nm.text()))
            cb = _guard(QtWidgets.QComboBox())
            cb.addItem("own person", 0)
            for j in ids:
                if j != k:
                    cb.addItem("Same as %d" % j, j)
                    cb.setItemData(cb.count() - 1, self._pname(j, e), QtCore.Qt.ToolTipRole)
            cb.setItemText(0, "Own")
            cb.setFixedWidth(104)
            cb.setCurrentIndex(max(0, cb.findData(same.get(k, 0))))
            cb.setEnabled(k not in removed and not new)
            if new:
                c.setEnabled(False)
                c.setToolTip("not propagated yet: press Propagate")
                cb.setItemText(0, "New")
            h.addWidget(sw)
            h.addWidget(c)
            h.addWidget(nm, 1)
            h.addWidget(cb)
            c.toggled.connect(lambda on, k=k: self._people_changed(k, keep=on))
            cb.currentIndexChanged.connect(lambda _i, k=k, cb=cb: self._people_changed(k, same=cb.currentData()))
            self.people_rows.addWidget(row)
        self.people_rows.addStretch(1)
        self.rows_scroll.setFixedHeight(min(170, 34 * max(1, len(ids) + len(extra)) + 6))
        cur = self.fix_target.currentData()
        self.fix_target.clear()
        for k in list(ids) + extra:
            self.fix_target.addItem(self._pname(k, e) + (" (new)" if k in extra else ""), k)
        self.fix_target.addItem("New person", 0)
        self.fix_target.addItem("New object", -1)
        self.fix_target.setCurrentIndex(max(0, self.fix_target.findData(cur)))

    def _count_text(self, ks, e=None):
        """'2 people and 1 object', '1 object', '3 people'"""
        n_o = sum(1 for k in ks if self._kind(k, e)[0] == "object")
        n_p = len(ks) - n_o
        parts = []
        if n_p:
            parts.append("%d %s" % (n_p, "person" if n_p == 1 else "people"))
        if n_o:
            parts.append("%d %s" % (n_o, "object" if n_o == 1 else "objects"))
        return " and ".join(parts) or "nothing"

    def _find_label(self):
        p = (self.s.people_prompt or "person").strip()
        return "Find everyone" if p.lower() in ("person", "people") else "Find: %s" % p

    def _kind(self, k, e=None):
        e = e if e is not None else (self.job.people_edits() if self.job else {})
        done = self.job._read_done("people") if self.job and self._people_ready() else {}
        kinds = dict(done.get("kinds", {}), **e.get("kinds", {}))
        return kinds.get(str(k), "person"), done.get("found_as", {}).get(str(k))

    def _pname(self, k, e=None):
        e = e if e is not None else (self.job.people_edits() if self.job else {})
        kind, found = self._kind(k, e)
        if e.get("names", {}).get(str(k)):
            return e["names"][str(k)]
        if found and kind == "object":            # found by Find everyone as "car" -> "Car 3"
            return "%s %d" % (found.capitalize(), k)
        return ("Object %d" if kind == "object" else "Person %d") % k

    def _rename(self, k, name):
        if self.job is None:
            return
        e = self.job.people_edits()
        names = dict(e.get("names", {}))
        name = " ".join(name.split())[:40]
        if (names.get(str(k)) or "") == name:
            return
        if name:
            names[str(k)] = name
        else:
            names.pop(str(k), None)
        e["names"] = names
        self.job.save_people_edits(e)
        QtCore.QTimer.singleShot(0, self._update_status)     # rows rebuilt after the edit closes

    def _people_changed(self, k, keep=None, same=None):
        e = self.job.people_edits()
        removed = set(int(x) for x in e.get("removed", []))
        sm = {int(a): int(b) for a, b in e.get("same", {}).items()}
        if keep is not None:
            removed.discard(k) if keep else removed.add(k)
        if same is not None:
            sm.pop(k, None)
            if same:
                sm[k] = int(same)
        e["removed"], e["same"] = sorted(removed), {str(a): b for a, b in sm.items()}
        self.job.save_people_edits(e)
        self._update_status()
        self._show()

    def _plate_size(self):
        d = self.job._read_done("plate")
        return d.get("width", 1), d.get("height", 1)

    # ---- keyframes (Sammie-Roto style): per frame and person, left = this person, right = not
    def _applied_keys(self):
        try:
            return self.job.people_edits().get("keys", []) if self.job else []
        except Exception:
            return []

    def _keys(self, frame=None):
        """applied keyframes with your pending changes on top (None = a keyframe you cleared)"""
        out = {(int(k["frame"]), int(k["target"])): k for k in self._applied_keys()}
        out.update(self.pending)
        keys = [k for k in out.values() if k is not None and k.get("points")]
        return [k for k in keys if frame is None or int(k["frame"]) == frame]

    def _people_image(self, i):
        """plate + people coloured with their number (your review applied) + the clicks of this frame"""
        if not self._plate_ready():
            return None
        import cv2
        from .stages import people as PP
        pth = self.job.ppath("plate", i)
        idp = self.job.cpath("people", i, "png")
        ready = self._people_ready() and os.path.isfile(idp)
        keys = self._keys(i)
        if not os.path.isfile(pth) or (not ready and not keys):
            return None
        rgb = cv2.imread(pth)[..., ::-1]
        ids = self.job.people_ids(i) if ready else np.zeros(rgb.shape[:2], np.uint8)
        pv = self.previews.get(i)
        if pv and os.path.isfile(pv) and any(f == i for f, _ in self.pending):
            from .io import sequence as seq
            ids = PP.lut(self.job.people_edits())[seq.read_png(pv)]      # live clicks result, this frame
        img = PP.overlay(np.ascontiguousarray(rgb), ids)
        W, H = self._plate_size()
        sx, sy = img.shape[1] / float(W), img.shape[0] / float(H)
        r = max(5, img.shape[1] // 200)
        for key in keys:                           # dots in the colour of who they are for
            col = PP.color(key["target"])
            for x, y, l in key["points"]:
                c = (int(x * sx), int(y * sy))
                cv2.circle(img, c, r + 2, (20, 20, 22), -1, cv2.LINE_AA)
                if l:
                    cv2.circle(img, c, r, col, -1, cv2.LINE_AA)
                else:                              # "not this person": a red ring with a bar
                    cv2.circle(img, c, r, (255, 69, 58), max(2, r // 3), cv2.LINE_AA)
                    cv2.line(img, (c[0] - r, c[1]), (c[0] + r, c[1]), (255, 69, 58), max(2, r // 3), cv2.LINE_AA)
        return img

    def _fix_mode(self, on):
        if on and self.view != "people":
            self._set_view("people")
        if on and self.b_play.isChecked():
            self.b_play.setChecked(False)
        self.viewer.set_click_mode(on)
        if on:
            self._start_sam()
        self._update_clicks_ui()

    def _next_id(self):
        used = set(self.job._read_done("people").get("ids", [])) if self._people_ready() else set()
        used |= {int(k["target"]) for k in self._keys()}
        used |= {int(t) for _, t in self.pending}
        e = self.job.people_edits()               # numbers still named / unticked / merged from before
        used |= {int(k) for k in e.get("removed", [])}
        for d in (e.get("same", {}), e.get("names", {}), e.get("kinds", {})):
            used |= {int(k) for k in d}
        used |= {int(v) for v in e.get("same", {}).values()}
        return max(used | {0}) + 1

    def _view_click(self, fx, fy, button):
        if not self.b_fix.isChecked() or self.job is None:
            return
        W, H = self._plate_size()
        x, y = fx * W, fy * H
        i = min(self.frame, self.job.n - 1)
        target = self.fix_target.currentData() or 0
        if target <= 0:                            # New person / object: give them the next number now
            obj = target < 0
            target = self._next_id()
            if obj:
                e = self.job.people_edits()
                e["kinds"] = dict(e.get("kinds", {}), **{str(target): "object"})
                self.job.save_people_edits(e)
            self.fix_target.addItem(self._pname(target) + " (new)", target)
            self.fix_target.setCurrentIndex(self.fix_target.count() - 1)
        k = (i, int(target))
        key = self.pending.get(k)
        if key is None:
            applied = next((a for a in self._applied_keys() if (int(a["frame"]), int(a["target"])) == k), None)
            key = json.loads(json.dumps(applied)) if applied else {"frame": i, "target": int(target), "points": [],
                                                                     "base": self._save_base(i, int(target))}
            self.pending[k] = key
        key["points"].append([round(float(x), 1), round(float(y), 1), 1 if button == 1 else 0])
        self.history.append(k)
        self._request_preview(i)
        self._update_clicks_ui()
        self._show()

    def _save_base(self, frame, target):
        """the person's mask on this frame before your first click there: the clicks refine it"""
        if not self._people_ready():
            return None
        from .io import sequence as seq
        ids = seq.read_png(self.job.cpath("people", frame, "png"))
        m = ids == target
        if not m.any():
            return None
        rel = os.path.join("pta_people_keys", "f%05d_p%d_%d.png" % (frame, target, int(time.time() * 1000) % 100000))
        p = os.path.join(self.job.out, rel)
        os.makedirs(os.path.dirname(p), exist_ok=True)
        seq.write_png(p, (m * 255).astype(np.uint8))
        return rel.replace("\\", "/")

    def _clear_clicks(self):
        """undo the last click"""
        while self.history:
            k = self.history.pop()
            key = self.pending.get(k)
            if key and key["points"]:
                key["points"].pop()
                if not key["points"]:
                    applied = any((int(a["frame"]), int(a["target"])) == k for a in self._applied_keys())
                    if applied:
                        self.pending[k] = None     # back to nothing on this frame for them
                    else:
                        self.pending.pop(k)
                self._request_preview(k[0])
                break
        self._update_clicks_ui()
        self._show()

    def _clear_all(self):
        if self.job is None or self.runner is not None:
            return
        if QtWidgets.QMessageBox.question(self, "PlateToAOVs", "Remove everyone, your clicks, names and objects, "
                                          "and start the segmentation again?") != QtWidgets.QMessageBox.Yes:
            return
        was_on = self.b_fix.isChecked()
        self._stop_sam()
        self.pending, self.history = {}, []
        self.previews.clear()
        self.job.clear_people()
        self._people_sig = None
        self.b_fix.setChecked(False)
        self._refresh_job()
        self._set_view("plate")
        if was_on:
            self.b_fix.setChecked(True)

    def _clear_frame(self):
        """remove the clicks of the selected person on this frame"""
        if self.job is None:
            return
        i = min(self.frame, self.job.n - 1)
        target = self.fix_target.currentData() or 0
        k = (i, int(target))
        if any((int(a["frame"]), int(a["target"])) == k for a in self._applied_keys()):
            self.pending[k] = None
        else:
            self.pending.pop(k, None)
        self.history = [h for h in self.history if h != k]
        self._request_preview(i)
        self._update_clicks_ui()
        self._show()

    # ---- live clicks (SAM 2.1 kept loaded)
    def _start_sam(self):
        if self.sam is not None or not self._plate_ready():
            return
        if self._missing_needed("people"):
            self.b_fix.setChecked(False)
            self._models("Selecting people needs SAM 2.1 and SAM 3.1: download them first.")
            return
        self.sam = SamServer(self.s, self)
        self.sam.ready.connect(self._sam_ready)
        self.sam.preview.connect(self._sam_preview)
        self.sam.failed.connect(lambda m: (self.fix_hint.setText("Live preview failed: %s" % m[:160]),
                                           self.logwin.appendPlainText("[live clicks] " + m)))
        self.sam.log.connect(lambda m: self.logwin.appendPlainText("[live clicks] " + m))
        self.sam.start()
        self._update_clicks_ui()

    def _stop_sam(self):
        if self.sam is not None:
            self.sam.stop()
            self.sam.deleteLater()
            self.sam = None
        self._want.clear()

    def _sam_ready(self):
        for f in sorted(set(f for f, _ in self.pending)):
            self._request_preview(f)
        self._update_clicks_ui()

    def _request_preview(self, frame):
        if not any(f == frame for f, _ in self.pending):
            self.previews.pop(frame, None)
            self._want.pop(frame, None)
            return
        if self.sam is None or not self.sam.is_ready:
            return                                 # sent when SAM is ready
        self._want[frame] = self.sam.request(frame, self._keys(frame))
        self._update_clicks_ui()

    def _sam_preview(self, rid, frame, path):
        if self._want.get(frame) != rid:
            return                                 # an older click, a newer one is on its way
        self._want.pop(frame, None)
        if any(f == frame for f, _ in self.pending):
            self.previews[frame] = path
        self._update_clicks_ui()
        if self.job is not None and min(self.frame, self.job.n - 1) == frame:
            self._show()

    def _update_clicks_ui(self):
        n_frames = len(set(f for f, _ in self.pending))
        self.b_apply.setEnabled(bool(self.pending) and self.runner is None)
        self.b_clear.setEnabled(bool(self.history))
        on = self.b_fix.isChecked()
        self.fix_hint.setVisible(on or bool(self.pending))
        if on and self.sam is not None and not self.sam.is_ready:
            self.fix_hint.setText("Loading SAM 2.1 for live clicks (a few seconds)...")
        elif self._want:
            self.fix_hint.setText("Cutting...")
        elif self.pending:
            self.fix_hint.setText("Changes on %d frame%s. Propagate to follow everyone through the shot "
                                  "with your clicks added (the others stay as they are)."
                                  % (n_frames, "" if n_frames == 1 else "s"))
        else:
            self.fix_hint.setText("Pick who, then click. On someone already found: left click adds the part under "
                                  "it (a rifle...), right click removes it. New person: left = them, right = not them. "
                                  "Everyone found stays: Propagate adds your clicks to it.")

    def _apply_fixes(self):
        """Propagate: SAM 2.1 through the shot with all the keyframes"""
        if self.job is None:
            return
        e = self.job.people_edits()
        if not self._people_ready() and not e.get("keys"):
            e["auto"] = False                     # you selected the people yourself: only them
        keys = {(int(k["frame"]), int(k["target"])): k for k in e.get("keys", [])}
        keys.update(self.pending)
        e["keys"] = [k for k in keys.values() if k is not None and k.get("points")]
        e["validated"] = False
        self.job.save_people_edits(e)
        self.pending = {}
        self.history = []
        self.previews.clear()
        self._stop_sam()                          # frees the GPU for the shot run
        self._update_clicks_ui()
        self._run(until="people")

    # ---------------------------------------------------------------- files
    def _pick_plate(self):
        start = os.path.dirname(self.s.input_pattern) if self.s.input_pattern else os.path.expanduser("~")
        f, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Choose one frame of the plate", start,
                                                     "Images (*.exr *.png *.jpg *.jpeg *.tif *.tiff *.dpx)")
        if not f:
            return
        self.s.input_pattern = f
        self.s.output_dir = ""
        if f.lower().endswith(".exr") and _get(self.cs) == "srgb":
            _set(self.cs, "linear")
        elif f.lower().endswith((".png", ".jpg", ".jpeg")):
            _set(self.cs, "srgb")
        self._full = None                         # a new plate: its whole range
        self.s.first = self.s.last = 0
        self._refresh_job()

    def _pick_out(self):
        d = QtWidgets.QFileDialog.getExistingDirectory(self, "Output folder", self.job.out if self.job else "")
        if d:
            self.s.output_dir = d
            self._refresh_job()

    def _open_out(self):
        if self.job:
            os.makedirs(self.job.out, exist_ok=True)
            QtGui.QDesktopServices.openUrl(QtCore.QUrl.fromLocalFile(self.job.out))

    def closeEvent(self, e):
        self._stop_sam()
        if self.runner:
            if QtWidgets.QMessageBox.question(self, "PlateToAOVs", "Still generating. Stop and quit?") != QtWidgets.QMessageBox.Yes:
                return e.ignore()
            self.runner.cancel()
            self.runner.proc.waitForFinished(3000)
        self.logwin.close()
        e.accept()


def _short(p, n=40):
    p = p.replace("\\", "/")
    return p if len(p) <= n else "…" + p[-(n - 1):]


def _dark_palette():
    """dark Fusion palette: every widget (dialogs, menus, scroll areas) gets dark backgrounds,
    whatever the Windows theme is."""
    p = QtGui.QPalette()
    c = QtGui.QColor
    for role, col in ((QtGui.QPalette.Window, "#1c1c1e"), (QtGui.QPalette.WindowText, "#ececec"),
                      (QtGui.QPalette.Base, "#2c2c2e"), (QtGui.QPalette.AlternateBase, "#232325"),
                      (QtGui.QPalette.Text, "#ececec"), (QtGui.QPalette.Button, "#3a3a3c"),
                      (QtGui.QPalette.ButtonText, "#ececec"), (QtGui.QPalette.ToolTipBase, "#2c2c2e"),
                      (QtGui.QPalette.ToolTipText, "#ececec"), (QtGui.QPalette.Highlight, ACCENT),
                      (QtGui.QPalette.HighlightedText, "#ffffff"), (QtGui.QPalette.PlaceholderText, "#8e8e93")):
        p.setColor(role, c(col))
    for role in (QtGui.QPalette.WindowText, QtGui.QPalette.Text, QtGui.QPalette.ButtonText):
        p.setColor(QtGui.QPalette.Disabled, role, c("#636366"))
    return p


def main():
    if sys.platform == "win32":                 # own taskbar entry and icon, not python.exe's
        try:
            import ctypes
            ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID("PlateToAOVs.PTA")
        except Exception:
            pass
    app = QtWidgets.QApplication(sys.argv)
    app.setApplicationName("PlateToAOVs")
    ico = os.path.join(ASSETS, "pta_icon.ico")
    app.setWindowIcon(QtGui.QIcon(ico if os.path.isfile(ico) else os.path.join(ASSETS, "pta_icon.png")))
    app.setStyle("Fusion")
    app.setPalette(_dark_palette())
    app.setStyleSheet(STYLE)
    w = Main()
    w.show()
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
