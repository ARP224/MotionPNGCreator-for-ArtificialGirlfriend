@echo off
rem NOTE: Keep this file ASCII-only with CRLF line endings.
cd /d "%~dp0"
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0tools\install.ps1"
if errorlevel 1 (
    echo.
    echo Setup did not finish. Please check the messages above.
    pause
)
