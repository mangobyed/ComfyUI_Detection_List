#!/bin/bash

# ComfyUI YOLOv8 Detection Node Installation Script

echo "Installing ComfyUI YOLOv8 Detection Node..."

# Check if pip is available
if ! command -v pip &> /dev/null; then
    echo "Error: pip is not installed. Please install Python and pip first."
    exit 1
fi

# Install required packages
echo "Installing required Python packages..."
pip install -r requirements.txt

# Check if ultralytics is properly installed
python -c "import ultralytics; print('✓ ultralytics installed successfully')" 2>/dev/null || {
    echo "Error: Failed to install ultralytics. Trying alternative installation..."
    pip install ultralytics --upgrade
}

# Download default YOLOv8 model (optional)
echo "Downloading default YOLOv8 nano model..."
python -c "
from ultralytics import YOLO
try:
    model = YOLO('yolov8n.pt')
    print('✓ YOLOv8 nano model downloaded successfully')
except Exception as e:
    print(f'Warning: Could not download model: {e}')
    print('Model will be downloaded automatically on first use')
"

echo ""
echo "Installation completed!"
echo ""
echo "To use this node in ComfyUI:"
echo "1. Copy this folder to your ComfyUI/custom_nodes/ directory"
echo "2. Restart ComfyUI"
echo "3. Look for 'YOLOv8 Object Detection' in the image/object_detection category"
echo ""
echo "For more information, see the README.md file."
