@echo off
setlocal
cd /d "%~dp0"
"runtime\python_exact\python.exe" -s -B "build_reviewer_index.py"
exit /b %errorlevel%
