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

# Install PyTorch with CUDA support
echo ""
echo "Installing PyTorch with CUDA support..."
pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu130

# ============================================================================
# CUDA 13 / GB10 Blackwell Configuration
# ============================================================================
echo ""
echo "======================================"
echo "🔧 Configuring for GB10 Blackwell GPU"
echo "======================================"

# Check for CUDA 13
CUDA13_NVCC="/usr/local/cuda-13.0/bin/nvcc"
if [ -f "$CUDA13_NVCC" ]; then
    echo "✓ Found CUDA 13 compiler: $CUDA13_NVCC"
    
    echo ""
    echo "Installing llama-cpp-python with CUDA 13 support..."
    echo "This may take several minutes to compile..."
    echo ""
    
    # Build llama-cpp-python with correct CUDA 13 compiler for Blackwell
    CMAKE_ARGS="-DGGML_CUDA=on -DCMAKE_CUDA_COMPILER=$CUDA13_NVCC -DCMAKE_CUDA_ARCHITECTURES=120" \
        pip install llama-cpp-python --no-cache-dir --force-reinstall
    
    if [ $? -eq 0 ]; then
        echo "✓ llama-cpp-python installed with CUDA 13 / Blackwell support"
    else
        echo "❌ Failed to build llama-cpp-python with CUDA support"
        echo "Falling back to CPU-only version..."
        pip install llama-cpp-python
    fi
else
    echo "⚠️  CUDA 13 not found at $CUDA13_NVCC"
    echo "Checking for other CUDA installations..."
    
    # Try to find any CUDA installation
    if [ -d "/usr/local/cuda" ]; then
        CUDA_NVCC="/usr/local/cuda/bin/nvcc"
        if [ -f "$CUDA_NVCC" ]; then
            CUDA_VERSION=$($CUDA_NVCC --version | grep "release" | sed 's/.*release //' | sed 's/,.*//')
            echo "Found CUDA $CUDA_VERSION at $CUDA_NVCC"
            
            CMAKE_ARGS="-DGGML_CUDA=on -DCMAKE_CUDA_COMPILER=$CUDA_NVCC" \
                pip install llama-cpp-python --no-cache-dir --force-reinstall
        fi
    else
        echo "No CUDA found. Installing CPU-only version..."
        pip install llama-cpp-python
    fi
fi

# Verify llama-cpp-python installation
echo ""
echo "Verifying llama-cpp-python installation..."
if python3 -c "from llama_cpp import Llama; print('✓ llama-cpp-python OK')" 2>/dev/null; then
    echo "✓ llama-cpp-python installed successfully"
else
    echo "❌ llama-cpp-python installation failed"
    exit 1
fi
# ============================================================================
# ============================================================================

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
