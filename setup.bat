@echo off
setlocal
cd /d "%~dp0"
echo === PlateToAOVs setup ===
echo Needs: Windows, an NVIDIA GPU (24 GB recommended), git, and uv or Python 3.13.
echo.

rem ---- 1. the app's own Python environment (.venv), with uv inside it for the next steps
where git >nul 2>&1 || (echo git not found: install it from https://git-scm.com & goto :fail)
set "PY=%~dp0.venv\Scripts\python.exe"
set "UV=%~dp0.venv\Scripts\uv.exe"
if not exist "%PY%" (
  echo [1/5] creating .venv ^(Python 3.13^)
  where uv >nul 2>&1
  if not errorlevel 1 (
    uv venv .venv --python 3.13 || goto :fail
  ) else (
    py -3.13 -m venv .venv || (echo Install uv ^(https://docs.astral.sh/uv^) or Python 3.13, then run setup.bat again. & goto :fail)
  )
) else (
  echo [1/5] .venv already there
)
if not exist "%UV%" (
  where uv >nul 2>&1
  if not errorlevel 1 (
    uv pip install --python "%PY%" uv || goto :fail
  ) else (
    "%PY%" -m ensurepip >nul 2>&1
    "%PY%" -m pip install uv || goto :fail
  )
)

rem ---- 2. ComfyUI: yours if you have one (its models are reused), otherwise a copy in vendor\ComfyUI
echo [2/5] ComfyUI code ^(used as a library, nothing is changed in it^)
"%PY%" -m pta.setup_paths %* || goto :fail
for /f "usebackq delims=" %%a in (`"%PY%" -m pta.setup_paths --get`) do set "COMFY=%%a"

echo [3/5] torch 2.12.1 + CUDA 13.0
set "TORCH_IDX=https://download.pytorch.org/whl/cu130"
"%UV%" pip install --python "%PY%" torch==2.12.1+cu130 torchvision==0.27.1+cu130 torchaudio==2.11.0+cu130 --index-url %TORCH_IDX% || goto :fail
rem  pins so that nothing below swaps torch for a CPU build from PyPI
> constraints.txt echo torch==2.12.1+cu130
>> constraints.txt echo torchvision==0.27.1+cu130
>> constraints.txt echo torchaudio==2.11.0+cu130
set "PIN=--constraint constraints.txt --extra-index-url %TORCH_IDX% --index-strategy unsafe-best-match"

echo [4/5] app + ComfyUI code dependencies
"%UV%" pip install --python "%PY%" -r requirements.txt %PIN% || goto :fail
"%UV%" pip install --python "%PY%" -r "%COMFY%\requirements.txt" %PIN% || goto :fail

echo [5/5] model code into vendor\
if not exist "vendor\sapiens2\sapiens" (
  if exist "%COMFY%\custom_nodes\ComfyUI-Sapiens2-Easy\vendor\sapiens2\sapiens" (
    robocopy "%COMFY%\custom_nodes\ComfyUI-Sapiens2-Easy\vendor\sapiens2" "vendor\sapiens2" /E /XD __pycache__ .git /NFL /NDL /NJH /NJS >nul
  ) else (
    git clone --depth 1 https://github.com/facebookresearch/sapiens2.git vendor\sapiens2 || goto :fail
  )
)
if not exist "vendor\vda\video_depth_anything" (
  if exist "%COMFY%\custom_nodes\ComfyUI-Video-Depth-Anything\video_depth_anything" (
    robocopy "%COMFY%\custom_nodes\ComfyUI-Video-Depth-Anything\video_depth_anything" "vendor\vda\video_depth_anything" /E /XD __pycache__ /NFL /NDL /NJH /NJS >nul
  ) else (
    git clone --depth 1 https://github.com/DepthAnything/Video-Depth-Anything.git vendor\vda || goto :fail
  )
)
if not exist "vendor\sam2\sam2\build_sam.py" (
  echo   SAM 2.1 ^(people keyframes, Apache-2.0^)
  git clone --depth 1 https://github.com/facebookresearch/sam2.git vendor\sam2 || goto :fail
)
if not exist "vendor\MatAnyone2\matanyone2\inference\inference_core.py" (
  echo   MatAnyone 2 ^(alpha of your selection, S-Lab License: non-commercial^)
  git clone --depth 1 https://github.com/pq-yang/MatAnyone2.git vendor\MatAnyone2 || goto :fail
)
if not exist "vendor\materia\diffusion_renderer_pipeline.py" (
  echo   Materia ^(Cosmos DiffusionRenderer in PyTorch, GPL-3.0^)
  git clone --depth 1 https://github.com/NicholaiVogel/comfyui-materia.git vendor\materia || goto :fail
)

"%PY%" -c "import torch, PySide6; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())" || goto :fail
call make_shortcut.bat
echo.
echo Setup OK. Start the app with the PlateToAOVs shortcut ^(Desktop^) or run.bat: it offers to download the models you need.
pause
exit /b 0

:fail
echo.
echo Setup FAILED (see the error above).
pause
exit /b 1
