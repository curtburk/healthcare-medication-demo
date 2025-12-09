# Medical AI Drug Interaction Demo

### What This Demo Is
This demo is a fully functional AI-powered clinical decision support tool that analyzes drug interactions in real time. A user enters medications, and the system instantly returns professional-grade analysis including interaction severity, clinical mechanisms, monitoring recommendations, and alternative medication suggestions.
The demo runs entirely on a single HP ZGX Nano workstation. No cloud connection required. No data leaves the device.

### What It Proves to Customers
1. Enterprise AI runs locally on HP hardware.
The demo uses Mixtral-8x7B, a 47-billion parameter large language model, the same class of AI that powers cloud services costing thousands per month. Customers see this running on hardware they can own and control.
2. Sensitive data stays on-premises.
Healthcare organizations cannot send patient medication lists to external cloud APIs due to HIPAA and data governance requirements. This demo shows AI analysis happening entirely within the customer's environment.
3. Fine-tuning works.
The model has been trained on authoritative medical datasets (FDA drug labels, Stanford's TWOSIDES interaction database, PubMed clinical studies). Customers see that they can customize AI models for their specific domain and data, not just use generic chatbots.


### Why Healthcare / Drug Interactions
Healthcare is the ideal vertical for this demonstration because:

Data privacy requirements make cloud AI problematic
Domain expertise matters (generic AI gives generic answers)
The use case is immediately understandable (everyone takes medications)
Clinical decision support is a real, funded market
Regulatory requirements favor on-premises solutions

The demo translates directly to customer conversations about radiology AI, clinical documentation, medical coding, pathology analysis, and any healthcare AI application requiring data sovereignty.

## The Customer Conversation
Opening: "Let me show you what enterprise AI looks like when it runs entirely on your hardware."

During Demo: Enter common drug pairs (warfarin + aspirin, metformin + contrast dye) or use the complex patient scenarios with multiple medications. Point out the response quality and speed.

Key Messages:

"This model has 47 billion parameters, running locally, no cloud required"

"The training data came from FDA and NIH sources, not internet scraping"

"Your patient data never leaves this machine"


Closing: "This is one example. The same hardware and workflow applies to any AI application where your data cannot leave your environment."

## Table of Contents

1. [Overview](#overview)
2. [System Requirements](#system-requirements)
3. [Project Architecture](#project-architecture)
4. [Directory Structure](#directory-structure)
5. [Quick Start Guide](#quick-start-guide)
6. [Complete Installation](#complete-installation)
7. [Data Collection Pipeline](#data-collection-pipeline)
8. [Model Fine-Tuning](#model-fine-tuning)
9. [Running the Demo](#running-the-demo)
10. [API Reference](#api-reference)
11. [Configuration Reference](#configuration-reference)
12. [Troubleshooting](#troubleshooting)

---

## Overview

This demo application provides clinical decision support for analyzing drug-drug interactions using a fine-tuned Mixtral-8x7B model. The application demonstrates two analysis modes:

**Simple Mode**: Analyze interactions between two specific medications, providing severity ratings, clinical mechanisms, and monitoring recommendations.

**Complex Mode**: Comprehensive patient case analysis supporting multiple medications, patient demographics, medical conditions, and laboratory values to assess polypharmacy risks.

The fine-tuned model draws on authoritative medical datasets including TWOSIDES (Stanford SNAP), DailyMed (FDA/NIH), and PubMed Central clinical studies.

---

## System Requirements

### Hardware Requirements

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| GPU | NVIDIA GPU with 24GB VRAM | HP ZGX Nano with NVIDIA GB10 Grace Blackwell |
| System Memory | 32GB RAM | 64GB+ unified memory |
| Storage | 50GB free space | 100GB+ SSD |
| CPU | 8 cores | 16+ cores |

### Software Requirements

| Software | Version |
|----------|---------|
| Operating System | Ubuntu 22.04 LTS or Ubuntu 24.04 LTS |
| Python | 3.10 or higher |
| CUDA | 12.0 or higher |
| pip | Latest version |

### Verify CUDA Installation

Before proceeding, verify your CUDA installation using the included test script:

```bash
python3 test-cuda.py
```

Expected output should confirm CUDA availability, device name, and successful GPU memory allocation.

---

## Project Architecture

The demo follows a standard FastAPI backend with HTML frontend architecture:

```
+-------------------+     +-------------------+     +-------------------+
|                   |     |                   |     |                   |
|  HTML Frontend    |---->|  FastAPI Backend  |---->|  Quantized LLM    |
|  (index.html)     |     |  (main.py)        |     |  (llama.cpp)      |
|                   |     |                   |     |                   |
+-------------------+     +-------------------+     +-------------------+
```

**Frontend**: Single-page HTML/CSS/JavaScript application with responsive design for both simple two-drug queries and complex patient case analysis.

**Backend**: FastAPI server providing RESTful endpoints for drug interaction analysis, health checks, and sample query retrieval.

**Inference Engine**: llama.cpp with 4-bit quantized Mixtral model for efficient GPU inference.

---

## Directory Structure

```
healthcare-demo/
|-- backend/
|   |-- main.py                    # FastAPI application server
|   |-- requirements.txt           # Python dependencies
|
|-- frontend/
|   |-- index.html                 # Web interface
|   |-- hp_logo.png                # HP branding assets
|
|-- models/
|   |-- medical-mixtral-q4.gguf    # Quantized model file (after download)
|
|-- medical_data/
|   |-- drug_labels/               # DailyMed FDA drug labels
|   |-- interactions/              # TWOSIDES interaction database
|   |-- clinical_studies/          # PMC research articles
|   |-- training_data/             # Processed instruction dataset
|   |   |-- train.jsonl            # Training split
|   |   |-- val.jsonl              # Validation split
|   |-- processed/
|       |-- unified_medical.db     # Combined SQLite database
|
|-- scripts/
|   |-- download-medical-data.py   # Data collection script
|   |-- prepare-instruction-data.py # Training data preparation
|   |-- finetune-llm.py            # Model fine-tuning script
|   |-- test-mixtral-ft.py         # Fine-tuned model testing
|   |-- test-cuda.py               # CUDA verification
|
|-- install.sh                     # Installation script
|-- start_demo_remote.sh           # Remote startup script
|-- download_models.sh             # Model download script
|-- logs/                          # Application logs
```

---

## Quick Start Guide

For sales demonstrations where the model and dependencies are pre-installed:

```bash
# Navigate to the demo directory
cd /home/curtburk/Desktop/healthcare-demo

# Start the demo server
./start_demo_remote.sh
```

Access the demo at `http://localhost:8000` or via the IP address displayed in the terminal output.

---

## Complete Installation

### Step 1: Clone or Copy Demo Files

Ensure all demo files are present in the target directory:

```bash
mkdir -p /home/curtburk/Desktop/healthcare-demo
cd /home/curtburk/Desktop/healthcare-demo
```

**[CONFIGURATION REQUIRED]** Update the path above to match your target installation directory.

### Step 2: Run the Installation Script

```bash
chmod +x install.sh
./install.sh
```

The installation script performs the following actions:

1. Verifies Python 3 and pip installation
2. Creates a virtual environment named `medical-env`
3. Installs all required Python packages from `backend/requirements.txt`
4. Creates necessary directory structure (frontend, models, logs)
5. Moves frontend files to the appropriate location
6. Checks for model availability

### Step 3: Download the Quantized Model

If the model is not pre-installed, download it from S3:

```bash
chmod +x download_models.sh
./download_models.sh
```

**[CONFIGURATION REQUIRED]** The download script requires AWS CLI with valid credentials. Configure with `aws configure` if not already set up.

**Note**: The quantized model file is approximately 25GB. Download time varies based on network speed.

### Step 4: Verify Installation

Activate the virtual environment and verify the setup:

```bash
source medical-env/bin/activate
python3 -c "from llama_cpp import Llama; print('llama-cpp-python loaded successfully')"
```

---

## Data Collection Pipeline

The data collection pipeline gathers authoritative medical information from multiple sources for model fine-tuning.

### Data Sources

| Source | Description | Records |
|--------|-------------|---------|
| TWOSIDES | Drug-drug interaction network from Stanford SNAP | ~48,514 interactions |
| DailyMed | FDA-approved drug labels with interaction warnings | Variable |
| PubMed Central | Clinical studies on drug interactions | Variable |
| DDI Corpus | Annotated drug interaction sentences | Variable |

### Running Data Collection

```bash
# Activate virtual environment
source medical-env/bin/activate

# Collect all data sources
python3 download-medical-data.py --data-dir ./medical_data --sources all

# Or collect specific sources
python3 download-medical-data.py --sources twosides dailymed
```

**Available source options**: `twosides`, `dailymed`, `pmc`, `ddi`, `all`

### Data Collection Output

The script creates a unified SQLite database at `medical_data/processed/unified_medical.db` containing normalized interaction data from all sources.

Collection summary is saved to `medical_data/collection_summary.json`.

---

## Model Fine-Tuning

### Step 1: Prepare Instruction Dataset

Convert collected medical data into instruction-following format:

```bash
source medical-env/bin/activate
python3 prepare-instruction-data.py
```

This generates approximately 5,000 training samples covering:

- Drug-drug interaction analysis (~4,000-5,000 samples)
- FDA label summarization (~100-120 samples)
- Clinical study interpretation (~150 samples)
- Polypharmacy scenarios (~100 samples)
- Comprehensive clinical decision scenarios (~30 samples)

Output files:

- `medical_data/training_data/train.jsonl` - Training split (90%)
- `medical_data/training_data/val.jsonl` - Validation split (10%)

### Step 2: Fine-Tune the Model

The fine-tuning script uses 4-bit quantization with LoRA adapters for memory efficiency:

```bash
source medical-env/bin/activate
python3 finetune-llm.py
```

**Fine-Tuning Configuration**:

| Parameter | Value |
|-----------|-------|
| Base Model | mistralai/Mixtral-8x7B-Instruct-v0.1 |
| Quantization | 4-bit NF4 |
| LoRA Rank | 16 |
| LoRA Alpha | 32 |
| Target Modules | q_proj, v_proj, k_proj, o_proj, gate_proj, up_proj, down_proj |
| Max Sequence Length | 128 tokens |
| Batch Size | 1 (with gradient accumulation of 8) |
| Learning Rate | 2e-4 |
| Optimizer | paged_adamw_8bit |

**[CONFIGURATION REQUIRED]** Update the following paths in `finetune-llm.py` if your directory structure differs:

```python
config = {
    "train_file": "./medical_data/training_data/train.jsonl",
    "val_file": "./medical_data/training_data/val.jsonl",
    "output_dir": "./mixtral_medical_production",
    ...
}
```

### Step 3: Test the Fine-Tuned Model

Verify the fine-tuned model produces appropriate responses:

```bash
python3 test-mixtral-ft.py
```

The test script loads the base model with LoRA adapters and runs a sample drug interaction query.

**[CONFIGURATION REQUIRED]** Update the LoRA adapter path in `test-mixtral-ft.py` if different:

```python
model = PeftModel.from_pretrained(base_model, "./mixtral_medical_lora")
```

### Step 4: Quantize for Production

After fine-tuning, quantize the model to GGUF format for llama.cpp inference. This step typically requires llama.cpp's conversion tools (not included in this demo package).

---

## Running the Demo

### Local Access

```bash
cd /home/curtburk/Desktop/healthcare-demo
source medical-env/bin/activate
python3 backend/main.py
```

Access the demo at: `http://localhost:8000`

### Remote Access (Recommended for Demonstrations)

Use the remote startup script for access from other machines:

```bash
./start_demo_remote.sh
```

The script displays the server IP address and provides instructions for SSH tunnel setup.

**SSH Tunnel Method (Most Reliable)**:

From a Windows laptop, open PowerShell and run:

```powershell
ssh -L 8000:localhost:8000 curtburk@<SERVER_IP>
```

**[CONFIGURATION REQUIRED]** Replace `curtburk` with the appropriate username and `<SERVER_IP>` with the actual server IP address.

Then access the demo at: `http://localhost:8000`

**Direct Network Access**:

If on the same network, access directly at: `http://<SERVER_IP>:8000`

### Startup Notes

- Initial model loading takes 1-2 minutes
- The server binds to `0.0.0.0:8000` to allow remote connections
- Press Ctrl+C to stop the server

---

## API Reference

### Health Check

```
GET /api/health
```

Returns server status and model information.

**Response**:
```json
{
    "status": "healthy",
    "model_loaded": true,
    "model_type": "Fine-tuned Mixtral Medical (Quantized Q4)",
    "inference_engine": "llama.cpp"
}
```

### Simple Drug Interaction Analysis

```
POST /api/simple_interaction
```

Analyzes interaction between two drugs.

**Request Body**:
```json
{
    "drug1": "warfarin",
    "drug2": "aspirin"
}
```

**Response**:
```json
{
    "success": true,
    "drug1": "warfarin",
    "drug2": "aspirin",
    "analysis": "...",
    "inference_time": 2.34,
    "complexity_level": "simple"
}
```

### Complex Patient Analysis

```
POST /api/complex_interaction
```

Comprehensive analysis for patients with multiple medications and conditions.

**Request Body**:
```json
{
    "medications": ["warfarin", "aspirin", "omeprazole", "metoprolol"],
    "age": 78,
    "conditions": ["atrial fibrillation", "hypertension", "GERD"],
    "lab_values": {"INR": "3.2", "CrCl": "45 mL/min"},
    "additional_context": "Optional notes"
}
```

**Response**:
```json
{
    "success": true,
    "patient_summary": {...},
    "analysis": "...",
    "risk_score": 7.3,
    "risk_level": "High",
    "inference_time": 4.56,
    "complexity_level": "complex"
}
```

### Sample Queries

```
GET /api/sample_queries
```

Returns pre-configured example queries for demonstration purposes.

---

## Configuration Reference

### Paths Requiring Configuration

The following file paths are hardcoded and must be updated for different installations:

**main.py** (line ~38):
```python
model_path = "/home/curtburk/Desktop/healthcare-demo/models/medical-mixtral-q4.gguf"
```

**install.sh** (lines ~76, ~82):
```bash
MODEL_PATH="/home/curtburk/Desktop/healthcare-demo/mixtral_medical_merged"
LORA_PATH="/home/curtburk/Desktop/healthcare-demo/mixtral_medical_production"
```

**start_demo_remote.sh** (lines ~43, ~44):
```bash
MODEL_PATH="/home/curtburk/Desktop/healthcare-demo/mixtral_medical_merged"
LORA_PATH="/home/curtburk/Desktop/healthcare-demo/mixtral_medical_production"
```

**download_models.sh** (line ~10):
```bash
S3_BUCKET="s3://finetuning-demo-models"
```

### Model Configuration Options

Adjust inference parameters in `main.py`:

```python
llm = Llama(
    model_path=model_path,
    n_gpu_layers=-1,    # -1 = use all layers on GPU
    n_ctx=2048,         # Context window size
    n_batch=512,        # Batch size for prompt processing
    n_threads=8,        # CPU threads for non-GPU operations
    verbose=False       # Set True for debugging
)
```

### Generation Parameters

Modify in the API endpoint functions:

```python
output = llm(
    prompt,
    max_tokens=512,      # Maximum response length
    temperature=0.7,     # Creativity (0.0-1.0)
    top_p=0.9,          # Nucleus sampling
    repeat_penalty=1.1,  # Reduce repetition
    stop=["</s>", "[INST]"]
)
```

---

## Troubleshooting

### Virtual Environment Not Found

**Error**: "Virtual environment not found! Please run install.sh first"

**Solution**: Run the installation script:
```bash
./install.sh
```

### Model File Not Found

**Error**: Application fails to start or crashes immediately

**Solution**: Verify the model path in `main.py` matches the actual model location:
```bash
ls -la /home/curtburk/Desktop/healthcare-demo/models/
```

If the model is missing, run:
```bash
./download_models.sh
```

### CUDA Out of Memory

**Error**: "CUDA out of memory" during inference

**Solution**: 
1. Reduce `n_ctx` value in model initialization
2. Reduce `max_tokens` in generation parameters
3. Ensure no other GPU processes are running:
   ```bash
   nvidia-smi
   ```

### AWS Credentials Error

**Error**: "AWS credentials not configured" during model download

**Solution**: Configure AWS CLI:
```bash
aws configure
```

Enter your AWS Access Key ID, Secret Access Key, default region, and output format.

### Port Already in Use

**Error**: "Address already in use" when starting server

**Solution**: Kill existing process on port 8000:
```bash
sudo lsof -i :8000
sudo kill -9 <PID>
```

### Frontend Not Found

**Error**: "Frontend not found" when accessing web interface

**Solution**: Ensure index.html is in the frontend directory:
```bash
ls -la frontend/
# If missing, copy from root:
cp index.html frontend/
```

### Slow Inference Times

**Issue**: Analysis takes longer than expected (>30 seconds)

**Possible Causes**:
1. Model not fully loaded on GPU - check `nvidia-smi` for GPU memory usage
2. First query after startup is always slower (model warmup)
3. Complex queries with long context take more time

**Solutions**:
1. Verify all layers are on GPU: set `n_gpu_layers=-1`
2. Run a simple query first to warm up the model
3. Reduce `max_tokens` for faster responses

### Connection Refused (Remote Access)

**Error**: Cannot connect from remote machine

**Solutions**:
1. Verify the server is running and bound to 0.0.0.0:
   ```bash
   netstat -tlnp | grep 8000
   ```
2. Check firewall settings:
   ```bash
   sudo ufw status
   sudo ufw allow 8000
   ```
3. Use SSH tunnel method as described in [Running the Demo](#running-the-demo)

---

## Support

For issues specific to this demo, contact the HP ZGX Nano product team.

For general questions about the fine-tuning process or model architecture, refer to:
- Hugging Face Transformers documentation
- PEFT (Parameter-Efficient Fine-Tuning) documentation
- llama.cpp project documentation

---

## Appendix: Key Dependencies

| Package | Purpose |
|---------|---------|
| fastapi | Web framework for API server |
| uvicorn | ASGI server |
| llama-cpp-python | Quantized model inference |
| transformers | Model loading and tokenization |
| peft | LoRA adapter training |
| bitsandbytes | 4-bit quantization |
| torch | PyTorch deep learning framework |
| pandas | Data processing |
| sqlite3 | Medical database storage |

Full dependency list available in `backend/requirements.txt`.
