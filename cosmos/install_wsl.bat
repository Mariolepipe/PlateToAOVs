@echo off
rem  Step 1 of 2: installs WSL (Linux inside Windows). Needs admin, then a restart.
rem  If Windows refuses (error 14098 = Windows component store corrupted), it is repaired
rem  with DISM + SFC (built into Windows) and the install is tried again.
net session >nul 2>&1
if errorlevel 1 (
  echo Asking for administrator rights...
  powershell -NoProfile -Command "Start-Process -FilePath '%~f0' -Verb RunAs"
  exit /b
)

echo === 1. Enabling the two Windows features WSL needs ===
call :features
if not errorlevel 1 goto wsl

echo.
echo === Windows refused. Repairing the Windows component store (DISM, 10-30 min, internet needed) ===
DISM /Online /Cleanup-Image /RestoreHealth
echo.
echo === Checking Windows system files (SFC, 5-15 min) ===
sfc /scannow
echo.
echo === Trying the features again ===
call :features
if errorlevel 1 goto still_refused

:wsl
echo.
echo === 2. WSL itself ===
wsl --install --no-distribution
wsl --update
echo.
echo OK. RESTART the PC, then run setup_cosmos.bat (step 2).
pause
exit /b 0

:still_refused
echo.
echo Still refused. Restart the PC and run install_wsl.bat once more.
echo If it fails again, send the last lines of C:\Windows\Logs\DISM\dism.log
pause
exit /b 1

:features
rem  0 = done, 3010 = done (restart needed), anything else (e.g. 14098) = refused
dism /online /enable-feature /featurename:Microsoft-Windows-Subsystem-Linux /all /norestart
set "RC=%errorlevel%"
if not "%RC%"=="0" if not "%RC%"=="3010" exit /b 1
dism /online /enable-feature /featurename:VirtualMachinePlatform /all /norestart
set "RC=%errorlevel%"
if not "%RC%"=="0" if not "%RC%"=="3010" exit /b 1
exit /b 0
