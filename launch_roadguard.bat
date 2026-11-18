@echo off
title ROADGUARD AI Launcher
cd /d "%~dp0"
echo ========================================================
echo ROADGUARD AI - Road Accident Intelligence Platform
echo Starting Command Center...
echo ========================================================

REM Check if Python venv exists
if not exist ".venv\Scripts\python.exe" (
    echo [ERROR] Virtual environment not found in .venv!
    echo Please install dependencies first.
    pause
    exit /b 1
)

echo Checking server status...
netstat -ano | findstr :8501 >nul
if %errorlevel% equ 0 (
    echo [INFO] ROADGUARD AI server is already running on port 8501!
    echo Opening browser...
    start "" "http://localhost:8501"
    exit /b 0
)

echo Starting Streamlit background process...
start "" "http://localhost:8501"
".venv\Scripts\python.exe" -m streamlit run app.py --server.port 8501 --server.headless true

if %errorlevel% neq 0 (
    echo.
    echo [ERROR] Streamlit exited with an error code: %errorlevel%
    pause
)
