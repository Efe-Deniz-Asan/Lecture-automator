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

echo Starting Board Calibration Mode...
echo ==============================================
echo INSTRUCTIONS:
echo 1. Put NEON GREEN (or your chosen color) tape on the corners of the board.
echo 2. Point the camera at the board.
echo 3. You will see colored boxes around the detected boards.
echo 4. Press the NUMBER key (1, 2, 3...) of the board you want to track.
echo 5. Press 'a' to track ALL detected boards.
echo 6. Press 'q' to Cancel.
echo ==============================================

:: Activate Venv
if exist ".venv\Scripts\activate.bat" (
    call .venv\Scripts\activate.bat
)

:: Run Calibration (Defaulting to Neon Green, but we can change this)
python lecture_automator.py calibrate --color neon-green --source 0 --use-yolo

echo.
echo Calibration Done.
pause
