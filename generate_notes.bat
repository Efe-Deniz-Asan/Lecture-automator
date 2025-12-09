@echo off
echo Starting Lecture Automator (Generation Mode)...
echo ===============================================

:: 1. Activate Virtual Environment
if exist ".venv\Scripts\activate.bat" (
    call .venv\Scripts\activate.bat
) else (
    echo ERROR: Virtual environment not found.
    pause
    exit /b
)

:: 2. Run Generation Command
:: Auto-detects language and uses your Gemini Key. 
:: Switched to 'turbo' model (Fast & Accurate) and Auto-Device (GPU if available).
echo Generating Study Guide (Auto-Language) for LATEST Lecture...
py -3.11 lecture_automator.py generate --model turbo --device auto --latest

echo.
echo ===============================================
echo DONE! Check the 'output' folder.
echo ===============================================
pause
