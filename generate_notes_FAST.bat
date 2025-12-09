@echo off
echo Starting Lecture Automator (FAST Generation Mode)...
echo ====================================================

:: 1. Activate Virtual Environment
if exist ".venv\Scripts\activate.bat" (
    call .venv\Scripts\activate.bat
) else (
    echo ERROR: Virtual environment not found.
    pause
    exit /b
)

:: 2. Run Generation Command with --skip-transcription
:: Takes existing transcript.txt and just runs Gemini.
echo Skipping Transcription... Generating Study Guide only (LATEST Lecture)...
py -3.11 lecture_automator.py generate --skip-transcription --latest

echo.
echo ===============================================
echo DONE! Check 'output\study_guide.md'.
echo ===============================================
pause
