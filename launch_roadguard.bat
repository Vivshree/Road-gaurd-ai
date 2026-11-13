@echo off
title ROADGUARD AI Launcher
cd /d "d:\road accident project"
echo ========================================================
echo ROADGUARD AI - Road Accident Intelligence Platform
echo Starting Command Center...
echo ========================================================
start "" "http://localhost:8501"
".venv\Scripts\streamlit.exe" run app.py --server.port 8501 --server.headless true
if %errorlevel% neq 0 (
    echo.
    echo Streamlit exited or encountered an error.
    pause
)
