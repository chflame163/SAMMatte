@echo off
setlocal

set "ROOT=%~dp0"

if not exist "%ROOT%setup_windows.ps1" (
  echo [ERROR] setup_windows.ps1 not found next to this batch file.
  pause
  exit /b 1
)

powershell -NoProfile -ExecutionPolicy Bypass -File "%ROOT%setup_windows.ps1" %*
set "EXIT_CODE=%ERRORLEVEL%"

if not "%EXIT_CODE%"=="0" (
  echo.
  echo setup_windows exited with code %EXIT_CODE%.
  pause
)
exit /b %EXIT_CODE%
