@echo off
title IntelliProfile — Setup & Launch
cd /d "%~dp0"

echo ============================================
echo   IntelliProfile — Auto Setup & Launch
echo ============================================
echo.

:: Check Python
where python >nul 2>&1
if %errorlevel% neq 0 (
    echo [FAIL] Python not found. Please install Python 3.10+ from https://python.org
    pause
    exit /b 1
)
echo [OK] Python found: 
python --version

:: Check / install requirements
echo.
echo [1/4] Checking dependencies...
python -X utf8 -c "import flask, sklearn, pandas, numpy, joblib, yaml, flasgger, flask_cors" >nul 2>&1
if %errorlevel% neq 0 (
    echo [..] Installing missing packages...
    pip install -r requirements.txt
    if %errorlevel% neq 0 (
        echo [FAIL] pip install failed. Check your internet connection.
        pause
        exit /b 1
    )
    echo [OK] Dependencies installed
) else (
    echo [OK] All dependencies satisfied
)

:: Ensure models directory exists
echo.
echo [2/4] Checking model...
if not exist "models\model_v*.joblib" (
    echo [..] No saved model found. Training new model...
    python -X utf8 profiler_main.py
    if %errorlevel% neq 0 (
        echo [FAIL] Model training failed.
        pause
        exit /b 1
    )
    echo [OK] Model trained successfully
) else (
    echo [OK] Existing model found
)

:: Verify tests
echo.
echo [3/4] Running tests...
python -X utf8 test_profiler.py
if %errorlevel% neq 0 (
    echo [WARN] Some tests failed. Starting anyway...
)

:: Start Flask server
echo.
echo [4/4] Starting IntelliProfile server...
echo.
start "" http://localhost:5000
python -X utf8 app.py

pause
