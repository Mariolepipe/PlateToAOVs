@echo off
rem  Step 2 of 2: Ubuntu in WSL, then Cosmos DiffusionRenderer inside it (~30 GB of weights).
rem  Everything Linux lives in one virtual disk file in WSLDIR below (set LAA_WSL_DIR to put it on another drive).
setlocal
cd /d "%~dp0"
set "DISTRO=Ubuntu-22.04"
set "WSLDIR=%LAA_WSL_DIR%"
if "%WSLDIR%"=="" set "WSLDIR=%LOCALAPPDATA%\LiveActionAOVs\WSL\Ubuntu-22.04"

wsl --status >nul 2>&1
if errorlevel 1 (
  echo WSL is not installed yet: run install_wsl.bat first, then restart the PC.
  pause
  exit /b 1
)

wsl -l -q | findstr /x /c:"%DISTRO%" >nul 2>&1
if errorlevel 1 (
  rem  wsl -l -q prints UTF-16: findstr may miss it, so also try running it
  wsl -d %DISTRO% -- true >nul 2>&1
  if errorlevel 1 (
    echo === Installing %DISTRO% into %WSLDIR% ===
    if not exist "%WSLDIR%\.." mkdir "%WSLDIR%\.."
    wsl --install -d %DISTRO% --location "%WSLDIR%" --no-launch
    if errorlevel 1 (
      echo   --location not supported by this WSL: installing then moving it to %WSLDIR%
      wsl --install -d %DISTRO% --no-launch || goto :fail
      wsl --manage %DISTRO% --move "%WSLDIR%" || echo   could not move it, it stays in the default WSL folder
    )
    echo.
    echo === First start of Ubuntu: choose a Linux user name and password ^(remember the password^) ===
    wsl -d %DISTRO% -- echo Ubuntu ready
  )
)

echo === Installing Cosmos inside Ubuntu (long: about 30 GB to download) ===
wsl -d %DISTRO% --cd "%~dp0." -- bash ./setup_cosmos.sh || goto :fail
echo.
echo Done. In LiveActionAOVs: Albedo ^> Best ^(Cosmos^).
pause
exit /b 0

:fail
echo.
echo Setup stopped (see the error above). Fix it and run setup_cosmos.bat again: finished steps are skipped.
pause
exit /b 1
