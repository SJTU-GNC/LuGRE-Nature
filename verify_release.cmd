@echo off
setlocal
cd /d "%~dp0"
set "LUGRE_PYTHON=python"
if exist "runtime\python_exact\python.exe" set "LUGRE_PYTHON=runtime\python_exact\python.exe"
"%LUGRE_PYTHON%" -s -B "verify_release.py"
exit /b %errorlevel%
