@echo off
REM ComfyUI YOLOv8 Detection Node Installation Script for Windows

echo Installing ComfyUI YOLOv8 Detection Node...

REM Check if pip is available
pip --version >nul 2>&1
if %errorlevel% neq 0 (
    echo Error: pip is not installed. Please install Python and pip first.
    pause
    exit /b 1
)

REM Install required packages
echo Installing required Python packages...
pip install -r requirements.txt

if %errorlevel% neq 0 (
    echo Error: Failed to install some packages. Please check your Python environment.
    pause
    exit /b 1
)

REM Check if ultralytics is properly installed
python -c "import ultralytics; print('✓ ultralytics installed successfully')" 2>nul
if %errorlevel% neq 0 (
    echo Error: Failed to install ultralytics. Trying alternative installation...
    pip install ultralytics --upgrade
)

REM Download default YOLOv8 model (optional)
echo Downloading default YOLOv8 nano model...
python -c "from ultralytics import YOLO; model = YOLO('yolov8n.pt'); print('✓ YOLOv8 nano model downloaded successfully')" 2>nul
if %errorlevel% neq 0 (
    echo Warning: Could not download model. Model will be downloaded automatically on first use.
)

echo.
echo Installation completed!
echo.
echo To use this node in ComfyUI:
echo 1. Copy this folder to your ComfyUI/custom_nodes/ directory
echo 2. Restart ComfyUI
echo 3. Look for 'YOLOv8 Object Detection' in the image/object_detection category
echo.
echo For more information, see the README.md file.
echo.
pause
