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
