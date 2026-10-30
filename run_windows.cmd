@echo off
setlocal
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
  echo Run setup_windows.cmd first using 64-bit Python 3.12.
  exit /b 1
)
".venv\Scripts\python.exe" run_all.py %*
exit /b %errorlevel%
