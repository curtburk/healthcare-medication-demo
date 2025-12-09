#!/bin/bash

# Installation script for Medical AI Drug Interaction Demo
# This script sets up the Python environment and installs all dependencies

echo "================================================"
echo "Medical AI Drug Interaction Demo - Installation"
echo "================================================"
echo ""

# Check if Python 3 is installed
if ! command -v python3 &> /dev/null; then
    echo "✗ Python 3 is not installed. Please install Python 3.8 or higher."
    exit 1
fi

echo "✓ Python 3 found: $(python3 --version)"
echo ""

# Check if pip is installed
if ! command -v pip3 &> /dev/null; then
    echo "✗ pip3 is not installed. Please install pip."
    exit 1
fi

echo "✓ pip3 found"
echo ""

# Create a virtual environment (optional but recommended)
echo "Creating Python virtual environment..."
if [ ! -d "medical-env" ]; then
    python3 -m venv medical-env
    echo "✓ Virtual environment created"
else
    echo "✓ Virtual environment already exists"
fi
echo ""

# Activate virtual environment
echo "Activating virtual environment..."
source medical-env/bin/activate
echo "✓ Virtual environment activated"
echo ""

# Upgrade pip
echo "Upgrading pip..."
pip install --upgrade pip
echo ""

# Install required packages
echo "Installing required Python packages..."
echo "This may take several minutes..."
echo ""
pip install -r backend/requirements.txt

if [ $? -eq 0 ]; then
    echo ""
    echo "✓ All packages installed successfully"
else
    echo ""
    echo "✗ Package installation failed"
    exit 1
fi
echo ""

# Create necessary directories
echo "Creating directory structure..."
mkdir -p frontend
mkdir -p models
mkdir -p logs
echo "✓ Directories created"
echo ""

# Move frontend files to frontend directory
echo "Setting up frontend files..."
if [ -f "index.html" ]; then
    mv index.html frontend/
    echo "✓ Moved index.html to frontend/"
fi

if [ -f "hp_logo.png" ]; then
    mv hp_logo.png frontend/
    echo "✓ Moved hp_logo.png to frontend/"
fi
echo ""

# Check if model exists
echo "Checking for medical AI model..."
MODEL_PATH="/home/curtburk/Desktop/healthcare-demo/mixtral_medical_merged"
if [ -d "$MODEL_PATH" ]; then
    echo "✓ Medical model found at: $MODEL_PATH"
else
    echo "⚠ Medical model not found at: $MODEL_PATH"
    echo "  Checking for LoRA adapters as fallback..."
    LORA_PATH="/home/curtburk/Desktop/healthcare-demo/mixtral_medical_production"
    if [ -d "$LORA_PATH" ]; then
        echo "✓ LoRA adapters found at: $LORA_PATH"
        echo "  The application will use base model + LoRA adapters"
    else
        echo "✗ Neither merged model nor LoRA adapters found"
        echo "  Please ensure the model is available"
    fi
fi
echo ""

# Download models (if needed)
echo "Setting up models..."
if [ -f "download_models.sh" ]; then
    echo "Running model download script..."
    bash download_models.sh
fi
echo ""

echo "================================================"
echo "Installation Complete!"
echo "================================================"
echo ""
echo "To start the demo:"
echo "  1. Activate the virtual environment: source medical-env/bin/activate"
echo "  2. Run: python3 backend/main.py"
echo "  3. Open browser to: http://localhost:8000"
echo ""
echo "For remote access from Windows laptop (PREFERRED METHOD):"
echo "  Run: ./start_demo_remote.sh"
echo ""
echo "Model Configuration:"
echo "  - Primary: Merged model at $MODEL_PATH"
echo "  - Fallback: Base model + LoRA at /home/curtburk/Desktop/healthcare-demo/mixtral_medical_production"
echo ""
echo "================================================"
