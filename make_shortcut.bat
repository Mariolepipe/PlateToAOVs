@echo off
rem PlateToAOVs shortcut (PTA icon, no console window), in this folder and on your Desktop.
rem Paths come from where this file is: works wherever the repo was cloned.
cd /d "%~dp0"
if not exist ".venv\Scripts\pythonw.exe" (
  echo Run setup.bat first.
  pause
  exit /b 1
)
set "PTA_DIR=%~dp0"
set "PTA_DESK=1"
if "%PTA_FRESH_TEST%"=="1" set "PTA_DESK=0"
powershell -NoProfile -ExecutionPolicy Bypass -Command ^
  "$d = $env:PTA_DIR.TrimEnd('\'); $w = New-Object -ComObject WScript.Shell; $l = @((Join-Path $d 'PlateToAOVs.lnk'));" ^
  "if ($env:PTA_DESK -eq '1') { $l += (Join-Path ([Environment]::GetFolderPath('Desktop')) 'PlateToAOVs.lnk') };" ^
  "foreach ($p in $l) {" ^
  "  $s = $w.CreateShortcut($p); $s.TargetPath = (Join-Path $d '.venv\Scripts\pythonw.exe'); $s.Arguments = '-m pta';" ^
  "  $s.WorkingDirectory = $d; $s.IconLocation = (Join-Path $d 'pta\assets\pta_icon.ico'); $s.Description = 'PlateToAOVs (PTA)'; $s.Save(); Write-Host ('  shortcut: ' + $p) }"
if errorlevel 1 (
  echo Could not create the shortcut ^(run.bat works too^).
  exit /b 1
)
exit /b 0
