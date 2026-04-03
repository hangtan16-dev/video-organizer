@echo off
echo ============================================
echo   Video Organizer - Install Dependencies
echo ============================================
echo.

where python >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python not found. Please install Python 3.10+ from https://python.org
    pause
    exit /b 1
)

echo Installing dependencies...
pip install -r requirements.txt

if errorlevel 1 (
    echo.
    echo ERROR: Failed to install dependencies.
    echo Try running as Administrator or using: pip install --user -r requirements.txt
    pause
    exit /b 1
)

echo.
echo Installation complete! Run "run.bat" to start the app.
pause
