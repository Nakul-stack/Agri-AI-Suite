@echo off
echo ============================================================
echo Starting Crop Recommendation System - Full Stack Server
echo ============================================================
echo.
echo Make sure to stop any existing server first (Ctrl+C if running)
echo.
timeout /t 3 /nobreak >nul
echo Starting server...
python server.py
pause

