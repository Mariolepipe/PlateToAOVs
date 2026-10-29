import os
import sys


def _log_file():
    """pythonw.exe (run.bat) has no console: what would be printed goes to a log file instead"""
    d = os.path.join(os.environ.get("LOCALAPPDATA") or os.path.expanduser("~"), "LiveActionAOVs")
    os.makedirs(d, exist_ok=True)
    return os.path.join(d, "laa.log")


if sys.stdout is None or sys.stderr is None:
    log = open(_log_file(), "w", encoding="utf-8", buffering=1)
    sys.stdout = sys.stdout or log
    sys.stderr = sys.stderr or log

try:
    from .ui import main
    code = main()
except SystemExit:
    raise
except BaseException:
    import traceback
    text = traceback.format_exc()
    sys.stderr.write(text)
    try:                                       # no console to read it in: show it
        from PySide6 import QtWidgets
        app = QtWidgets.QApplication.instance() or QtWidgets.QApplication(sys.argv)
        QtWidgets.QMessageBox.critical(None, "LiveActionAOVs", "LAA could not start:\n\n" + text[-2000:]
                                       + "\n\nrun_debug.bat shows the whole log.")
    except Exception:
        pass
    code = 1
sys.exit(code)
