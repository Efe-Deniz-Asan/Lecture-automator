@echo off
REM ------------------------------------------------------------------------------
REM  Copyright (c) 2025 Efe Deniz Asan
REM
REM  This program is free software: you can redistribute it and/or modify
REM  it under the terms of the GNU Affero General Public License as published
REM  by the Free Software Foundation, either version 3 of the License, or
REM  (at your option) any later version.
REM
REM  This program is distributed in the hope that it will be useful,
REM  but WITHOUT ANY WARRANTY; without even the implied warranty of
REM  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
REM  GNU Affero General Public License for more details.
REM
REM  You should have received a copy of the GNU Affero General Public License
REM  along with this program.  If not, see <https://www.gnu.org/licenses/>.
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
