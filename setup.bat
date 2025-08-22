@echo off
echo ================================
echo AI Object Segmentation Suite Setup
echo ================================

REM Check Python
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo ERROR: Python not found! Please install Python 3.8+
    pause
    exit /b 1
)

echo Found Python

REM Create virtual environment
echo.
echo Creating virtual environment...
if not exist "venv" (
    python -m venv venv
    echo Virtual environment created
) else (
    echo Virtual environment already exists
)

REM Activate virtual environment
echo.
echo Activating virtual environment...
call venv\Scripts\activate.bat

REM Upgrade pip
echo.
echo Upgrading pip...
python -m pip install --upgrade pip

REM Install requirements
echo.
echo Installing requirements...
echo This may take a few minutes...
pip install -r requirements.txt

REM Create output directory
echo.
echo Creating output directories...
if not exist "output" mkdir output

echo.
echo ================================
echo Setup Complete!
echo ================================
echo.
echo To get started:
echo 1. Activate environment: venv\Scripts\activate
echo 2. Test with an image: python segment_practical.py your_image.jpg
echo 3. Check results in: practical_output\
echo.
echo For more options, see README.md
pause
