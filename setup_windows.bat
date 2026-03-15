@echo off
echo ============================================
echo  Smart Farming DSS - Windows Setup Script
echo ============================================
echo.

REM Check Python
python --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Python not found. Please install Python 3.10+ from https://www.python.org/downloads/
    echo         Make sure to check "Add Python to PATH" during installation.
    pause
    exit /b 1
)
echo [OK] Python found.

REM Create virtual environment
echo.
echo Creating virtual environment...
python -m venv venv
if errorlevel 1 (
    echo [ERROR] Failed to create virtual environment.
    pause
    exit /b 1
)
echo [OK] Virtual environment created.

REM Activate and install
echo.
echo Installing dependencies...
call venv\Scripts\activate
pip install fastapi uvicorn[standard] python-multipart pydantic scikit-learn numpy pandas httpx edge-tts --quiet
if errorlevel 1 (
    echo [ERROR] Failed to install dependencies.
    pause
    exit /b 1
)
echo [OK] All dependencies installed.

REM Start server
echo.
echo ============================================
echo  Starting Smart Farming DSS server...
echo  Open http://localhost:8000 in your browser
echo  Press Ctrl+C to stop the server
echo ============================================
echo.
cd backend
python main.py
