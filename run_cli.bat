@echo off
rem  Batch mode, no window: Find finds the people (no clicks), then every pass that is on.
rem    run_cli.bat --input "D:\shots\030\plate.####.exr" --colorspace acescg --set alpha_source=matting
rem    run_cli.bat --list-settings        every setting, with its default
rem  Double-clicked (no arguments): asks for the plate.
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
  echo Run setup.bat first.
  pause
  exit /b 1
)
if not "%~1"=="" goto :run
echo PlateToAOVs, batch mode ^(no window^). From a terminal:
echo   run_cli.bat --input "C:\shots\plate.####.exr" --colorspace acescg
echo   run_cli.bat --list-settings
echo.
set "PLATE="
set /p "PLATE=Drag one frame of the plate here, then press Enter: "
if not defined PLATE exit /b 0
set "CS="
set /p "CS=Color space: srgb, linear or acescg [srgb]: "
if not defined CS set "CS=srgb"
".venv\Scripts\python.exe" -m pta.cli --input %PLATE% --colorspace %CS%
echo.
pause
exit /b
:run
".venv\Scripts\python.exe" -m pta.cli %*
if errorlevel 1 pause
