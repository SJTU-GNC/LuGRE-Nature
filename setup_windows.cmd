@echo off
setlocal
cd /d "%~dp0"
py -3.12 -c "import sys,struct;sys.exit(0 if sys.version_info[:2]==(3,12) and struct.calcsize('P')==8 else 1)" >nul 2>&1
if not errorlevel 1 goto use_launcher
python -c "import sys,struct;sys.exit(0 if sys.version_info[:2]==(3,12) and struct.calcsize('P')==8 else 1)" >nul 2>&1
if errorlevel 1 (
  echo Install 64-bit Python 3.12, then run this script again.
  exit /b 1
)
set "LUGRE_PYTHON=python"
goto create_environment
:use_launcher
set "LUGRE_PYTHON=py -3.12"
:create_environment
%LUGRE_PYTHON% -m venv .venv
if errorlevel 1 exit /b 1
if exist "offline_wheels" (
  ".venv\Scripts\python.exe" -m pip install --no-index --find-links "offline_wheels" -r requirements.txt
) else (
  ".venv\Scripts\python.exe" -m pip install -r requirements.txt
)
if errorlevel 1 exit /b 1
".venv\Scripts\python.exe" environment\check_environment.py
exit /b %errorlevel%
