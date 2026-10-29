@echo off
cd /d "%~dp0"
if not exist ".venv\Scripts\pythonw.exe" (
  echo Run setup.bat first.
  pause
  exit /b 1
)
rem pythonw: no black window. run_debug.bat shows everything in a console.
start "" ".venv\Scripts\pythonw.exe" -m laa %*
