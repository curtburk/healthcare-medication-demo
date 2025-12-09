#!/bin/bash

# Medical AI Demo - Model Download Script
# Downloads the quantized medical Mixtral model from S3

set -e  # Exit on error

echo "================================================"
echo "Medical AI Model Download Script"
echo "================================================"

# Configuration
S3_BUCKET="s3://finetuning-demo-models"
MODEL_NAME="medical-mixtral-q4.gguf"
MODEL_PATH="medical-mixtral-q4-quantized"
LOCAL_DIR="./models"

# Create models directory if it doesn't exist
mkdir -p $LOCAL_DIR

# Check if model already exists
if [ -f "$LOCAL_DIR/$MODEL_NAME" ]; then
    echo "✓ Model already exists at $LOCAL_DIR/$MODEL_NAME"
    echo "  To re-download, delete the existing file first"
    exit 0
fi

# Check if AWS CLI is installed
if ! command -v aws &> /dev/null; then
    echo "❌ AWS CLI is not installed. Please install it first:"
    echo "   sudo apt-get install awscli"
    exit 1
fi

# Check AWS credentials
echo "Checking AWS credentials..."
if ! aws sts get-caller-identity &> /dev/null; then
    echo "❌ AWS credentials not configured. Run 'aws configure' first"
    exit 1
fi

# Download the quantized model
echo "Downloading quantized medical model..."
echo "This is a 25GB file and may take several minutes..."

aws s3 cp $S3_BUCKET/$MODEL_PATH/$MODEL_NAME $LOCAL_DIR/$MODEL_NAME \
    --no-progress 2>&1 | while read line; do
    echo -n "."
done

echo ""

# Verify download
if [ -f "$LOCAL_DIR/$MODEL_NAME" ]; then
    FILE_SIZE=$(ls -lh "$LOCAL_DIR/$MODEL_NAME" | awk '{print $5}')
    echo "✓ Model downloaded successfully!"
    echo "  Location: $LOCAL_DIR/$MODEL_NAME"
    echo "  Size: $FILE_SIZE"
    
    # Update the model path in main.py if needed
    if [ -f "backend/main.py" ]; then
        echo "Updating backend/main.py with correct model path..."
        sed -i "s|model_path = .*|model_path = \"$LOCAL_DIR/$MODEL_NAME\"|g" backend/main.py
        echo "✓ Backend configuration updated"
    fi
else
    echo "❌ Download failed!"
    exit 1
fi

echo ""
echo "================================================"
echo "Model ready for use!"
echo "Start the demo with: ./start_demo_remote.sh"
echo "================================================"
