#!/bin/bash

# Download fine-tuned medical model from S3 bucket
echo "================================================"
echo "Downloading Fine-Tuned Medical Model from S3"
echo "================================================"
echo ""

# Configuration
S3_MODEL_URI="s3://finetuning-demo-models/finetuned-mistral-medical-MoE-7x8B/"
LOCAL_MODEL_DIR="models/finetuned-mistral-medical-MoE-7x8B"

# Create the models directory if it doesn't exist
mkdir -p models

# Check if AWS CLI is installed
if ! command -v aws &> /dev/null; then
    echo "⚠ AWS CLI not found. Attempting to install..."
    pip install awscli
    
    if [ $? -ne 0 ]; then
        echo "✗ Failed to install AWS CLI"
        echo "  Please install manually: pip install awscli"
        exit 1
    fi
fi

echo "Model Source: $S3_MODEL_URI"
echo "Local Destination: $LOCAL_MODEL_DIR"
echo ""

# Check if model already exists locally
if [ -d "$LOCAL_MODEL_DIR" ]; then
    echo "⚠ Model directory already exists locally"
    echo -n "  Do you want to re-download? (y/n): "
    read -r response
    
    if [ "$response" != "y" ]; then
        echo "  Skipping download, using existing model"
        exit 0
    else
        echo "  Removing existing model directory..."
        rm -rf "$LOCAL_MODEL_DIR"
    fi
fi

echo "Starting download..."
echo "This may take several minutes depending on your connection speed..."
echo ""

# Download the model from S3
aws s3 cp "$S3_MODEL_URI" "$LOCAL_MODEL_DIR/" --recursive --no-sign-request

# Check if download was successful
if [ $? -eq 0 ]; then
    echo ""
    echo "✓ Model downloaded successfully!"
    
    # Check the size of downloaded model
    if [ -d "$LOCAL_MODEL_DIR" ]; then
        MODEL_SIZE=$(du -sh "$LOCAL_MODEL_DIR" | cut -f1)
        echo "✓ Model size: $MODEL_SIZE"
        
        # List key model files
        echo ""
        echo "Model files:"
        ls -lh "$LOCAL_MODEL_DIR" | head -10
    fi
else
    echo ""
    echo "✗ Failed to download model from S3"
    echo ""
    echo "Troubleshooting tips:"
    echo "  1. Check your internet connection"
    echo "  2. Verify AWS CLI is configured correctly"
    echo "  3. Ensure the S3 bucket is publicly accessible"
    echo "  4. Try adding AWS credentials if the bucket is private:"
    echo "     aws configure"
    exit 1
fi

echo ""
echo "================================================"
echo "Model Download Complete!"
echo "================================================"
echo ""
echo "Model Location: $LOCAL_MODEL_DIR"
echo ""
echo "Next steps:"
echo "  1. Update backend/main.py to use this model path"
echo "  2. Run ./install.sh to set up the environment"
echo "  3. Run ./start_demo_remote.sh to start the demo"
echo ""
echo "================================================"