#!/bin/bash

# Setup script for AI Object Segmentation Suite
echo "================================"
echo "AI Object Segmentation Suite Setup"
echo "================================"

# Check Python version
python_version=$(python3 --version 2>&1 | grep -Po '(?<=Python )\d+\.\d+')
echo "✓ Found Python $python_version"

# Create virtual environment
echo ""
echo "Creating virtual environment..."
if [ ! -d "venv" ]; then
    python3 -m venv venv
    echo "✓ Virtual environment created"
else
    echo "✓ Virtual environment already exists"
fi

# Activate virtual environment
echo ""
echo "Activating virtual environment..."
source venv/bin/activate

# Upgrade pip
echo ""
echo "Upgrading pip..."
pip install --upgrade pip --quiet

# Install requirements
echo ""
echo "Installing requirements..."
echo "This may take a few minutes..."
pip install -r requirements.txt

# Download sample image for testing
echo ""
echo "Downloading sample test image..."
if [ ! -f "sample_test.jpg" ]; then
    curl -s -o sample_test.jpg "https://images.unsplash.com/photo-1529156069898-49953e39b3ac?w=800" 
    echo "✓ Sample image downloaded as 'sample_test.jpg'"
else
    echo "✓ Sample image already exists"
fi

# Create output directories
echo ""
echo "Creating output directories..."
mkdir -p output

echo ""
echo "================================"
echo "Setup Complete!"
echo "================================"
echo ""
echo "To get started:"
echo "1. Activate the environment: source venv/bin/activate"
echo "2. Test with sample image: python segment_practical.py sample_test.jpg"
echo "3. Check results in: practical_output/"
echo ""
echo "For more options, see README.md"
