@echo off
rem  MoGe speed test, with and without torch.inference_mode(), on a shot already run by the app.
rem  Usage: bench_moge.bat "C:\path\to\aovs_folder" [frames]
cd /d "%~dp0"
if "%~1"=="" (
  echo Usage: bench_moge.bat "C:\path\to\aovs_folder" [frames]
  pause
  exit /b 1
)
set "N=%~2"
if "%N%"=="" set "N=12"
".venv\Scripts\python.exe" -m laa.bench_moge "%~1" --frames %N%
pause
