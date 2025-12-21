@echo off
REM ------------------------------------------------------------------------------
REM  Copyright (c) 2025 Efe Deniz Asan
REM  All Rights Reserved.
REM
REM  NOTICE:  All information contained herein is, and remains the property of
REM  Efe Deniz Asan. The intellectual and technical concepts contained herein
REM  are proprietary to Efe Deniz Asan and are protected by trade secret or
REM  copyright law. Dissemination of this information or reproduction of this
REM  material is strictly forbidden unless prior written permission is obtained
REM  from Efe Deniz Asan.
REM ------------------------------------------------------------------------------

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
python lecture_automator.py generate --model turbo --device auto --latest

echo.
echo ===============================================
echo DONE! Check the 'output' folder.
echo ===============================================
pause
