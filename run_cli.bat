@echo off
rem  Batch mode, no UI.  Example:
rem    run_cli.bat --input "D:\shots\030\plate.####.exr" --colorspace acescg --set alpha_source=matting
cd /d "%~dp0"
".venv\Scripts\python.exe" -m laa.cli %*
