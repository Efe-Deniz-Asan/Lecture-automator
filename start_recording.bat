@echo off
REM ------------------------------------------------------------------------------
REM  Copyright (c) 2025 Efe Deniz Asan <asan.efe.deniz@gmail.com>
REM  All Rights Reserved.
REM
REM  NOTICE:  All information contained herein is, and remains the property of
REM  Efe Deniz Asan. The intellectual and technical concepts contained herein
REM  are proprietary to Efe Deniz Asan and are protected by trade secret or
REM  copyright law. Dissemination of this information or reproduction of this
REM  material is strictly forbidden unless prior written permission is obtained
REM  from Efe Deniz Asan or via email at <asan.efe.deniz@gmail.com>.
REM ------------------------------------------------------------------------------

echo Starting Lecture Automator (Recording Mode)...
echo ==============================================

:: 1. Activate Virtual Environment
if exist ".venv\Scripts\activate.bat" (
    call .venv\Scripts\activate.bat
) else (
    echo ERROR: Virtual environment not found. Please run 'python -m venv .venv' first.
    pause
    exit /b
)

:: 2. Run Recording Command
:: Using your settings: Mic ID 2, Chalk-Red mode, Hybrid YOLO Reliability
:: 2. Run Recording Command (Infinite Loop for Reliability)
:: If the app crashes (Camera disconnected?), it waits 5 seconds and restarts.
:loop
echo Running with Configured Audio Device, Red Chalk, and Hybrid Vision...
python lecture_automator.py record --color chalk-red --use-yolo

echo.
echo WARNING: App crashed or stopped. Restarting in 5 seconds...
timeout /t 5
goto loop

echo.
echo Recording finished.
pause
