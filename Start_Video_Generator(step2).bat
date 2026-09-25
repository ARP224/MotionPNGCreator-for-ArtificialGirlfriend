@echo off
rem NOTE: Keep this file ASCII-only with CRLF line endings.
cd /d "%~dp0"

rem Normal launch: no console window. Launched via launch_hidden.vbs because
rem uv-created venv pythonw.exe can be a console-subsystem trampoline that
rem would otherwise open a console window.
rem To see logs, run manually from a terminal:
rem     uv run python video_generator.py
if exist ".venv\Scripts\pythonw.exe" (
    wscript //nologo "tools\launch_hidden.vbs" ".venv\Scripts\pythonw.exe" "video_generator.py"
    exit /b
)

rem Fallback: the venv is not built yet. Run in this console via uv so the
rem first-time environment setup (uv sync) progress is visible.
set "UV=uv"
where uv >nul 2>nul
if errorlevel 1 (
    if exist "%USERPROFILE%\.local\bin\uv.exe" (
        set "UV=%USERPROFILE%\.local\bin\uv.exe"
    ) else (
        echo [ERROR] uv was not found.
        echo         Please run "Installer MotionPNGCreator for AG.bat" first.
        pause
        exit /b 1
    )
)

"%UV%" run python video_generator.py
if errorlevel 1 (
    echo.
    echo [ERROR] The program exited with an error. Please check the messages above.
    pause
)
