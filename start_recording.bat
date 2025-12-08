@echo off
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
