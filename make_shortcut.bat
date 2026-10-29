@echo off
rem LiveActionAOVs shortcut (LAA icon, no console window), in this folder and on your Desktop.
rem Paths come from where this file is: works wherever the repo was cloned.
cd /d "%~dp0"
if not exist ".venv\Scripts\pythonw.exe" (
  echo Run setup.bat first.
  pause
  exit /b 1
)
set "LAA_DIR=%~dp0"
set "LAA_DESK=1"
if "%LAA_FRESH_TEST%"=="1" set "LAA_DESK=0"
powershell -NoProfile -ExecutionPolicy Bypass -Command ^
  "$d = $env:LAA_DIR.TrimEnd('\'); $w = New-Object -ComObject WScript.Shell; $l = @((Join-Path $d 'LiveActionAOVs.lnk'));" ^
  "if ($env:LAA_DESK -eq '1') { $l += (Join-Path ([Environment]::GetFolderPath('Desktop')) 'LiveActionAOVs.lnk') };" ^
  "foreach ($p in $l) {" ^
  "  $s = $w.CreateShortcut($p); $s.TargetPath = (Join-Path $d '.venv\Scripts\pythonw.exe'); $s.Arguments = '-m laa';" ^
  "  $s.WorkingDirectory = $d; $s.IconLocation = (Join-Path $d 'laa\assets\laa_icon.ico'); $s.Description = 'LiveActionAOVs (LAA)'; $s.Save(); Write-Host ('  shortcut: ' + $p) }"
if errorlevel 1 (
  echo Could not create the shortcut ^(run.bat works too^).
  exit /b 1
)
exit /b 0
