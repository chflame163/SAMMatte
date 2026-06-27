@echo off
setlocal

set "ROOT=%~dp0"
set "APP=%ROOT%app"
set "MODELS=%ROOT%models"

where python >nul 2>nul
if errorlevel 1 (
  echo [ERROR] Python was not found.
  echo         Make sure the external Python environment is installed and "python" is available in PATH.
  pause
  exit /b 1
)
set "PY=python"

if "%SAM31_HOST%"=="" set "SAM31_HOST=127.0.0.1"
if "%SAM31_PORT%"=="" set "SAM31_PORT=8765"

if not exist "%APP%\run_sam31_webapp.py" (
  echo [ERROR] App entry not found: "%APP%\run_sam31_webapp.py"
  pause
  exit /b 1
)

if not exist "%MODELS%\sam3" (
  echo [ERROR] Missing SAM 3 source repo: "%MODELS%\sam3"
  echo         See README.md for model and repository setup instructions.
  pause
  exit /b 1
)

if not exist "%MODELS%\sam3.1\sam3.1_multiplex.pt" (
  echo [ERROR] Missing SAM 3.1 checkpoint: "%MODELS%\sam3.1\sam3.1_multiplex.pt"
  echo         See README.md for model and repository setup instructions.
  pause
  exit /b 1
)

if not exist "%MODELS%\vitmatte-base-composition-1k" (
  echo [WARN] ViTMatte directory was not found: "%MODELS%\vitmatte-base-composition-1k"
  echo        ViTMatte postprocess will not be available until the model is downloaded.
)

if not exist "%MODELS%\VideoMaMa" (
  echo [WARN] VideoMaMa directory was not found: "%MODELS%\VideoMaMa"
  echo        VideoMaMa postprocess will not be available until the repo and weights are downloaded.
)

set "PYTHONPATH=%APP%;%MODELS%\sam3;%PYTHONPATH%"
set "PATH=%ROOT%tools\ffmpeg;%PATH%"

"%PY%" -c "import pkg_resources, setuptools; print('[runtime] setuptools=' + setuptools.__version__ + ' pkg_resources_ok=True')"
if errorlevel 1 (
  echo.
  echo [ERROR] Missing setuptools/pkg_resources. Run: pip install setuptools
  pause
  exit /b 1
)

"%PY%" -c "import torch; print('[runtime] torch=' + torch.__version__ + ' cuda_runtime=' + str(torch.version.cuda) + ' cuda_available=' + str(torch.cuda.is_available())); raise SystemExit(0 if torch.cuda.is_available() else 2)"
if errorlevel 1 (
  echo.
  echo [ERROR] CUDA is not available to PyTorch. Check your NVIDIA driver and PyTorch install.
  pause
  exit /b 1
)

where ffmpeg >nul 2>nul
if errorlevel 1 (
  echo [WARN] ffmpeg.exe was not found. Preview or export H.264 re-encoding may fail.
  echo        Install ffmpeg into PATH or put ffmpeg.exe under "%ROOT%tools\ffmpeg".
)

"%PY%" "%APP%\run_sam31_webapp.py" --host %SAM31_HOST% --port %SAM31_PORT% --open-browser %*
set "EXIT_CODE=%ERRORLEVEL%"
if not "%EXIT_CODE%"=="0" (
  echo.
  echo SAMMatte exited with code %EXIT_CODE%.
  pause
)
exit /b %EXIT_CODE%
