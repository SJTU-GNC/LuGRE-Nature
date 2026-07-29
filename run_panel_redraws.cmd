@echo off
setlocal
cd /d "%~dp0"
call "run_all.cmd" --group panel %*
exit /b %errorlevel%
