@echo off
setlocal
cd /d "%~dp0"
"runtime\python_exact\python.exe" -s -B "build_file_hashes.py"
exit /b %errorlevel%
