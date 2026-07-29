@echo off
setlocal
cd /d "%~dp0"
call "run_all.cmd" --group exact %*
exit /b %errorlevel%
