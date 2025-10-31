@echo off
REM UniMate Chatbot Setup & Start Script for Windows

echo ========================================
echo   UniMate Chatbot Setup ^& Launcher
echo ========================================
echo.

REM Check if Python is installed
python --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Python is not installed or not in PATH
    echo Please install Python 3.8 or higher from python.org
    pause
    exit /b 1
)

echo [OK] Python is installed
echo.

REM Check if .env exists
if not exist .env (
    echo [WARNING] .env file not found!
    echo Creating .env template...
    (
        echo GOOGLE_API_KEY=your_gemini_api_key_here
        echo GEMINI_MODEL=gemini-2.5-flash
    ) > .env
    echo [OK] .env file created!
    echo.
    echo [IMPORTANT] Please edit .env and add your GOOGLE_API_KEY
    echo.
    pause
)

REM Check if API key is set
findstr /C:"your_gemini_api_key_here" .env >nul
if not errorlevel 1 (
    echo [ERROR] GOOGLE_API_KEY not configured in .env file!
    echo Please edit .env and add your Gemini API key
    pause
    exit /b 1
)

echo [OK] API key configured
echo.

REM Check if requirements are installed
echo Checking dependencies...
python -c "import flask" >nul 2>&1
if errorlevel 1 (
    echo [WARNING] Flask not found. Installing dependencies...
    if exist requirements_api.txt (
        pip install -r requirements_api.txt
    ) else (
        echo [ERROR] requirements_api.txt not found!
        pause
        exit /b 1
    )
)

echo [OK] All dependencies installed
echo.

REM Create necessary directories
echo Creating directories...
if not exist backend\vector_store mkdir backend\vector_store
if not exist backend\history mkdir backend\history
if not exist data\uploaded_files mkdir data\uploaded_files
echo [OK] Directories created
echo.

REM Start API server
echo ========================================
echo   Starting API Server...
echo ========================================
echo.

start "UniMate API Server" python api_server.py

REM Wait for API to start
timeout /t 3 /nobreak >nul

echo.
echo ========================================
echo   UniMate Chatbot is Ready!
echo ========================================
echo.
echo [API] http://localhost:5000
echo [TEST] Open test.html in your browser
echo.
echo Integration: Add the chatbot snippet to your website
echo See 'embed_snippet.html' for the code
echo.
echo API Endpoints:
echo   - POST /api/session/create
echo   - POST /api/upload
echo   - POST /api/query
echo   - GET  /api/health
echo.
echo Press any key to stop the server...
pause >nul

REM Kill the API server
taskkill /F /FI "WINDOWTITLE eq UniMate API Server*" >nul 2>&1
echo.
echo [OK] Server stopped
pause