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
