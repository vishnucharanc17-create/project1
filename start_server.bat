@echo off
REM ========================================
REM  Search Agent - Quick Start Script
REM ========================================

echo.
echo ========================================
echo   Search Agent Dashboard Launcher
echo ========================================
echo.

REM Check if virtual environment exists
if exist "venv\Scripts\activate.bat" (
    echo [1/3] Activating virtual environment...
    call venv\Scripts\activate.bat
) else (
    echo [!] Warning: Virtual environment not found
    echo [!] Using system Python
)

echo.
echo [2/3] Starting FastAPI server on http://localhost:8000
echo.
echo Press Ctrl+C to stop the server
echo.

REM Start server
python -m uvicorn app.main:app --reload --host 0.0.0.0 --port 8000

pause
