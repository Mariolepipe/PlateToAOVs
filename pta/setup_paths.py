"""
Run by setup.bat: finds ComfyUI (only its code is used, as a library) and the models folders,
or downloads ComfyUI into vendor/ComfyUI, then writes config.json. Standard library only.

    python -m pta.setup_paths            interactive
    python -m pta.setup_paths --yes      take what was found (or download ComfyUI) without asking
    python -m pta.setup_paths --get      print the ComfyUI folder from config.json (for setup.bat)
"""

import json
import os
import re
import subprocess
import sys

from . import config

APP = str(config.APP_DIR)
VENDOR_COMFY = os.path.join(APP, "vendor", "ComfyUI")


def is_comfy(d):
    return bool(d) and os.path.isfile(os.path.join(d, "comfy", "model_management.py"))


def fresh_test():
    """PTA_FRESH_TEST=1: behave as on a PC without ComfyUI (to test a first install)"""
    return os.environ.get("PTA_FRESH_TEST") == "1"


def desktop_app():
    return os.path.join(os.environ.get("LOCALAPPDATA", ""), "Programs", "@comfyorgcomfyui-electron", "resources", "ComfyUI")


def candidates():
    """Likely ComfyUI code folders on Windows: desktop app, portable build, git clones."""
    if fresh_test():
        return []
    env = os.environ.get
    home = os.path.expanduser("~")
    roots = [env("PTA_COMFYUI", ""),
             desktop_app(),
             os.path.join(home, "Documents", "ComfyUI"), os.path.join(home, "ComfyUI")]
    for drive in "CDEFG":
        base = drive + ":\\"
        if not os.path.isdir(base):
            continue
        roots += [base + "ComfyUI", base + "ComfyUI_windows_portable\\ComfyUI",
                  base + "AI\\ComfyUI", base + "AI\\ComfyUI_windows_portable\\ComfyUI"]
        try:                                   # one level of "<drive>:\<something>\ComfyUI..."
            for name in os.listdir(base):
                if "comfy" in name.lower():
                    p = os.path.join(base, name)
                    roots += [p, os.path.join(p, "ComfyUI"), os.path.join(p, "ComfyUI_windows_portable", "ComfyUI")]
                    if os.path.isdir(os.path.join(p, "ComfyUI-Installs")):        # some launchers
                        for inst in os.listdir(os.path.join(p, "ComfyUI-Installs")):
                            roots.append(os.path.join(p, "ComfyUI-Installs", inst, "ComfyUI"))
        except OSError:
            pass
    for dl in ("Downloads", "Desktop"):
        roots += [os.path.join(home, dl, "ComfyUI_windows_portable", "ComfyUI"), os.path.join(home, dl, "ComfyUI")]
    seen, out = set(), []
    for r in roots:
        r = os.path.normpath(r) if r else r
        if r and r.lower() not in seen and is_comfy(r):
            seen.add(r.lower())
            out.append(r)
    return out


def model_dirs(comfy):
    """ComfyUI's models folder + any extra folders from its extra_model_paths.yaml files."""
    dirs = [os.path.join(comfy, "models")]
    yamls = [os.path.join(comfy, "extra_model_paths.yaml")]
    if os.path.normcase(os.path.normpath(comfy)) == os.path.normcase(os.path.normpath(desktop_app())):
        yamls.append(os.path.join(os.environ.get("APPDATA", ""), "ComfyUI", "extra_models_config.yaml"))
    for y in yamls:
        if not os.path.isfile(y):
            continue
        try:
            with open(y, "r", encoding="utf-8") as f:
                for line in f:
                    m = re.match(r"\s*base_path\s*:\s*['\"]?(.+?)['\"]?\s*$", line)
                    if m:
                        b = os.path.expandvars(os.path.expanduser(m.group(1)))
                        b = b if os.path.isabs(b) else os.path.join(os.path.dirname(y), b)
                        for c in (os.path.join(b, "models"), b):
                            if os.path.isdir(c):
                                dirs.append(os.path.normpath(c))
                                break
        except OSError:
            pass
    dirs.append("models")
    out = []
    for d in dirs:
        if d not in out and (d == "models" or os.path.isdir(d)):
            out.append(d)
    return out


def clone():
    if is_comfy(VENDOR_COMFY):
        return VENDOR_COMFY
    print("Downloading ComfyUI %s into vendor\\ComfyUI (only its code is used)..." % config.COMFYUI_TAG)
    subprocess.check_call(["git", "clone", "--depth", "1", "--branch", config.COMFYUI_TAG,
                           "https://github.com/Comfy-Org/ComfyUI.git", VENDOR_COMFY])
    return VENDOR_COMFY


def write(comfy):
    data = {}
    if config.CONFIG_PATH.is_file():
        with open(config.CONFIG_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
    rel = os.path.relpath(comfy, APP) if os.path.normcase(comfy).startswith(os.path.normcase(APP)) else comfy
    data["comfyui_dir"] = rel.replace("\\", "/")
    keep = [d for d in data.get("model_dirs", []) if d != "models" and os.path.isdir(config._abs(d))]
    data["model_dirs"] = [d.replace("\\", "/") for d in dict.fromkeys(keep + model_dirs(comfy))]
    data.setdefault("sapiens2_repo", "vendor/sapiens2")
    data.setdefault("vda_code", "vendor/vda")
    with open(config.CONFIG_PATH, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
    print("config.json:")
    print("  ComfyUI code : %s" % data["comfyui_dir"])
    for d in data["model_dirs"]:
        print("  models       : %s" % d)


def ask(found):
    if found:
        print("ComfyUI found:")
        for i, p in enumerate(found):
            print("  [%d] %s" % (i + 1, p))
    else:
        print("No ComfyUI found on this PC.")
    print("Enter a number, the path of another ComfyUI folder, or D to download one into vendor\\ComfyUI.")
    while True:
        a = input("Choice [%s]: " % ("1" if found else "D")).strip().strip('"') or ("1" if found else "D")
        if a.lower() == "d":
            return clone()
        if a.isdigit() and 1 <= int(a) <= len(found):
            return found[int(a) - 1]
        for p in (a, os.path.join(a, "ComfyUI")):
            if is_comfy(p):
                return os.path.normpath(p)
        print("  not a ComfyUI folder (no comfy\\model_management.py inside): %s" % a)


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    if "--get" in argv:
        print(config.load_paths()["comfyui_dir"])
        return 0
    current = config.load_paths()["comfyui_dir"]
    if config.CONFIG_PATH.is_file() and is_comfy(current):
        print("ComfyUI already set up: %s" % current)
        write(current)                         # refresh the model folders
        return 0
    found = candidates()
    comfy = (found[0] if found else clone()) if "--yes" in argv else ask(found)
    write(comfy)
    return 0


if __name__ == "__main__":
    sys.exit(main())
