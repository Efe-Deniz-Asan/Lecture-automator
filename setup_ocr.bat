@echo off
echo ============================================
echo Math OCR Setup for Lecture Automator
echo ============================================
echo.

:: 1. Check if Tesseract is installed
echo [1/3] Checking Tesseract installation...
where tesseract >nul 2>&1
if %ERRORLEVEL% EQU 0 (
    echo [OK] Tesseract is already installed
    tesseract --version | findstr "tesseract"
) else (
    echo [!] Tesseract not found
    echo.
    echo Installing Tesseract-OCR via winget...
    winget install --id UB-Mannheim.TesseractOCR --accept-package-agreements --accept-source-agreements
    
    echo.
    echo [!] IMPORTANT: You may need to restart your terminal
    echo     or add Tesseract to PATH manually.
    echo     Default location: C:\Program Files\Tesseract-OCR
    pause
)

echo.
echo [2/3] Activating virtual environment...
if exist ".venv\Scripts\activate.bat" (
    call .venv\Scripts\activate.bat
) else (
    echo ERROR: Virtual environment not found.
    pause
    exit /b
)

echo.
echo [3/3] Installing Python OCR packages...
echo This may take a few minutes (downloading Pix2Tex model ~200MB)...
python -m pip install --upgrade pip
python -m pip install pytesseract pix2tex Pillow

echo.
echo ============================================
echo Setup Complete!
echo ============================================
echo.
echo Testing OCR availability...
python -c "import pytesseract; print('[OK] pytesseract installed')" 2>nul && echo [OK] Pytesseract || echo [FAIL] Pytesseract
python -c "import pix2tex; print('[OK] pix2tex installed')" 2>nul && echo [OK] Pix2Tex || echo [FAIL] Pix2Tex
python -c "from PIL import Image; print('[OK] Pillow installed')" 2>nul && echo [OK] Pillow || echo [FAIL] Pillow

echo.
echo ============================================
echo Next Steps:
echo 1. Record a lecture with equations on board
echo 2. Run generate_notes.bat
echo 3. Check study_guide.md for LaTeX equations
echo ============================================
pause
