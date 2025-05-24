@echo off
echo Starting Gemini to OpenAI Proxy Server...

REM Check Python
python --version >nul 2>&1
IF %ERRORLEVEL% NEQ 0 (
    echo ERROR: Python not found. Install Python. Add to PATH.
    pause
    exit /b 1
)

REM Navigate
cd /d "%~dp0"

REM Virtual environment
IF NOT EXIST "venv\Scripts\activate.bat" (
    echo Creating venv...
    python -m venv venv
    IF %ERRORLEVEL% NEQ 0 (
        echo ERROR: Failed venv creation.
        pause
        exit /b 1
    )
    call .\venv\Scripts\activate.bat
    echo Installing requirements...
    python -m pip install -r requirements.txt
    IF %ERRORLEVEL% NEQ 0 (
        echo ERROR: Failed requirements install.
        pause
        exit /b 1
    )
    echo Setup done.
) ELSE (
    call .\venv\Scripts\activate.bat
)

REM Check .env
IF NOT EXIST ".env" (
    echo.
    echo WARNING: .env file not found.
    echo Create .env with GEMINI_API_KEY.
    echo See README.
    echo.
    pause
    exit /b 1
)

echo Launching server...
python run.py

echo Server stopped or failed.
pause