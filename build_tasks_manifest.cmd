@echo off
setlocal
cd /d "%~dp0"
"runtime\python_exact\python.exe" -s -B "build_tasks_manifest.py"
exit /b %errorlevel%
