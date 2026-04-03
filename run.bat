@echo off
cd /d "%~dp0"
python main.py
if errorlevel 1 (
    echo.
    echo Error running the application. Make sure dependencies are installed.
    echo Run install.bat first.
    pause
)
