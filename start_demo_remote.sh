#!/bin/bash

# Script to start the Medical AI demo on HP ZGX Nano
# with access from a Windows laptop via SSH tunnel

echo "================================================"
echo "Medical AI Drug Interaction Demo - Remote Start"
echo "================================================"
echo ""

# Get the hostname/IP of the Linux server
HOSTNAME=$(hostname -I | awk '{print $1}')

# Update this to your ZGX's actual IP if needed
SERVER_IP="${HOSTNAME}"

echo "Server Information:"
echo "  Hostname/IP: $SERVER_IP"
echo "  Port: 8000"
echo ""

# Check if virtual environment exists
if [ ! -d "medical-env" ]; then
    echo "✗ Virtual environment not found!"
    echo "  Please run install.sh first"
    exit 1
fi

# Activate virtual environment
echo "Activating virtual environment..."
source medical-env/bin/activate
echo "✓ Virtual environment activated"
echo ""

# Check if main.py exists
if [ ! -f "backend/main.py" ]; then
    echo "✗ backend/main.py not found!"
    exit 1
fi

# Check model availability
echo "Checking model availability..."
MODEL_PATH="/home/curtburk/Desktop/healthcare-demo/mixtral_medical_merged"
LORA_PATH="/home/curtburk/Desktop/healthcare-demo/mixtral_medical_production"

if [ -d "$MODEL_PATH" ]; then
    echo "✓ Using merged medical model"
elif [ -d "$LORA_PATH" ]; then
    echo "✓ Using base model + LoRA adapters"
else
    echo "⚠ Warning: Model not found, the application may fail to start"
fi
echo ""

echo "================================================"
echo "Starting Medical AI Demo Server..."
echo "================================================"
echo ""
echo "Server will be accessible at:"
echo "  Local:  http://localhost:8000"
echo "  Remote: http://${SERVER_IP}:8000"
echo ""
echo "For Windows laptop access via SSH tunnel:"
echo "  1. Open PowerShell or Command Prompt on your Windows laptop"
echo "  2. Run: ssh -L 8000:localhost:8000 curtburk@${SERVER_IP}"
echo "  3. Then open browser to: http://localhost:8000"
echo ""
echo "Direct network access (if on same network):"
echo "  Open browser to: http://${SERVER_IP}:8000"
echo ""
echo "The model will take 1-2 minutes to load on first startup..."
echo "Press Ctrl+C to stop the server"
echo "================================================"
echo ""

# Start the FastAPI server
python3 backend/main.py

# Note: Uvicorn will bind to 0.0.0.0:8000 which allows remote connections
