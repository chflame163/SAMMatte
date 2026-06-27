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

set "PYTHONPATH=%APP%;%MODELS%\sam3;%PYTHONPATH%"
set "PATH=%ROOT%tools\ffmpeg;%PATH%"

"%PY%" -c "import sys, os, torch, torchvision, cv2, numpy, timm, transformers, PIL, iopath, huggingface_hub, pkg_resources, setuptools; print('python', sys.version); print('python_exe', sys.executable); print('torch', torch.__version__, 'cuda_runtime', torch.version.cuda, 'cuda_available', torch.cuda.is_available()); print('torchvision', torchvision.__version__); print('cv2', cv2.__version__); print('numpy', numpy.__version__); print('timm', timm.__version__); print('transformers', transformers.__version__); print('Pillow', PIL.__version__); print('setuptools', setuptools.__version__); print('pkg_resources_ok', hasattr(pkg_resources, 'resource_filename')); print('sam3_repo_exists', os.path.isdir(r'%MODELS%\\sam3')); print('sam31_checkpoint_exists', os.path.isfile(r'%MODELS%\\sam3.1\\sam3.1_multiplex.pt')); print('vitmatte_exists', os.path.isdir(r'%MODELS%\\vitmatte-base-composition-1k')); print('videomama_repo_exists', os.path.isdir(r'%MODELS%\\VideoMaMa'));"
if errorlevel 1 (
  echo.
  echo [ERROR] Runtime check failed.
  pause
  exit /b 1
)

where ffmpeg >nul 2>nul
if errorlevel 1 (
  echo.
  echo [WARN] ffmpeg.exe was not found in PATH.
  echo        Preview/export H.264 re-encoding may fail until ffmpeg is installed.
)

echo.
echo Runtime check finished.
pause
