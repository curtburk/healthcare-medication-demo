from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
import warnings
import logging
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import time
import os

warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- Model Path Definitions ---
MODEL_PATH = "/home/curtburk/Desktop/healthcare-demo/models/finetuned-mistral-medical-MoE-7x8B"
# Define a separate cache path for the 4-bit weights
# The model will be saved here AFTER the first time it is quantized.
QUANTIZED_MODEL_CACHE_PATH = "/home/curtburk/Desktop/healthcare-demo/models/finetuned-mistral-medical-MoE-7x8B_4bit_cache"
# ------------------------------

app = FastAPI(title="Medical AI Drug Interaction Demo")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files for frontend
app.mount("/static", StaticFiles(directory="frontend"), name="static")

# Request models
class SimpleDrugQuery(BaseModel):
    drug1: str
    drug2: str

class ComplexDrugQuery(BaseModel):
    medications: List[str]
    age: int
    conditions: List[str]
    lab_values: Optional[dict] = {}
    additional_context: Optional[str] = ""

# Global model variables
model = None
tokenizer = None
model_loaded = False

def load_model():
    """Load the model, prioritizing the cached 4-bit weights."""
    global model, tokenizer, model_loaded
    
    logger.info(f"Loading medical model. Checking cache path: {QUANTIZED_MODEL_CACHE_PATH}")
    
    try:
        # --- Define Base Quantization Config (Used for both loading and saving) ---
        bnb_config_base = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4", 
            bnb_4bit_compute_dtype=torch.float16, 
            bnb_4bit_use_double_quant=True,
        )

        # --- 1. Check if the 4-bit CACHE EXISTS ---
        if os.path.exists(QUANTIZED_MODEL_CACHE_PATH) and os.path.exists(os.path.join(QUANTIZED_MODEL_CACHE_PATH, "config.json")):
            
            logger.info("Found 4-bit cached model. Loading from cache...")
            
            # Load the model directly from the saved 4-bit cache path
            model = AutoModelForCausalLM.from_pretrained(
                QUANTIZED_MODEL_CACHE_PATH,
                quantization_config=bnb_config_base, # Still pass the config for correct loading behavior
                device_map="auto",
                trust_remote_code=True,
                low_cpu_mem_usage=True,
                attn_implementation ="flash_attention_2"
            )
            
        # --- 2. If NO cache, load and QUANTIZE/SAVE ---
        else:
            logger.warning(f"4-bit cache NOT found at {QUANTIZED_MODEL_CACHE_PATH}. Loading and quantizing (this will take time)...")

            # Load the model from the original path and perform quantization
            model = AutoModelForCausalLM.from_pretrained(
                MODEL_PATH,
                dtype=torch.float16,
                quantization_config=bnb_config_base, 
                device_map="auto",
                trust_remote_code=True,
                low_cpu_mem_usage=True,
                attn_implementation="flash_attention_2"
            )

            # --- CRITICAL CACHING STEP ---
            logger.info("Quantization complete. Saving 4-bit weights to cache for faster startup next time.")
            model.save_pretrained(QUANTIZED_MODEL_CACHE_PATH)
            # ---------------------------

        # --- 3. Load Tokenizer ---
        logger.info("Loading slow tokenizer...")
        # Use the original path for the tokenizer, as it's just config files
        tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, use_fast=False) 
        
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        # Set model to evaluation mode for faster inference
        model.eval()
        
        model_loaded = True
        logger.info("✓ Fine-tuned medical model loaded successfully!")
        
        # Log GPU status
        if torch.cuda.is_available():
            logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
            allocated = torch.cuda.memory_allocated() / 1024**3
            reserved = torch.cuda.memory_reserved() / 1024**3
            logger.info(f"GPU memory - Allocated: {allocated:.2f} GB, Reserved: {reserved:.2f} GB")
        
    except FileNotFoundError:
        logger.error(f"Error: Model not found at {MODEL_PATH} or {QUANTIZED_MODEL_CACHE_PATH}")
        model_loaded = False
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        import traceback
        logger.error(traceback.format_exc())
        model_loaded = False

# Load model on startup
load_model()

@app.get("/", response_class=HTMLResponse)
async def read_root():
    """Serve the main HTML page"""
    try:
        with open("frontend/index.html", "r") as f:
            return HTMLResponse(content=f.read())
    except FileNotFoundError:
        return HTMLResponse(content="<h1>Frontend not found.</h1>", status_code=500)

@app.post("/api/simple_interaction")
async def simple_drug_interaction(query: SimpleDrugQuery):
    """Check interaction between two drugs"""
    if not model_loaded:
        raise HTTPException(status_code=500, detail="Model not loaded")
    
    try:
        prompt = f"""[INST] As a clinical pharmacist, analyze the drug interaction between {query.drug1} and {query.drug2}.
        
        Provide:
        1. Interaction severity (None/Low/Moderate/High/Contraindicated)
        2. Clinical mechanism of interaction
        3. Patient monitoring recommendations
        4. Alternative medications if interaction is significant
        
        Be thorough but concise. [/INST]"""
        
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
        
        # Move to GPU
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
        
        start_time = time.time()
        
        # Generate with optimized settings
        with torch.cuda.amp.autocast():  # Mixed precision for speed
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=250,
                    temperature=0.7,
                    do_sample=True,
                    top_p=0.9,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                    use_cache=True,  # Enable KV cache
                )
        
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        # Extract only the response part
        if "[/INST]" in response:
            response = response.split("[/INST]")[-1].strip()
        
        inference_time = time.time() - start_time
        
        return JSONResponse(content={
            "success": True,
            "drug1": query.drug1,
            "drug2": query.drug2,
            "analysis": response,
            "inference_time": round(inference_time, 2),
            "complexity_level": "simple"
        })
        
    except Exception as e:
        logger.error(f"Error processing simple interaction: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/complex_interaction")
async def complex_drug_interaction(query: ComplexDrugQuery):
    """Analyze complex multi-drug interactions with patient factors"""
    if not model_loaded:
        raise HTTPException(status_code=500, detail="Model not loaded")
    
    try:
        # Build comprehensive patient context
        medications_str = ", ".join(query.medications)
        conditions_str = ", ".join(query.conditions) if query.conditions else "None reported"
        
        lab_context = ""
        if query.lab_values:
            lab_items = [f"{k}: {v}" for k, v in query.lab_values.items()]
            lab_context = f"Lab values: {', '.join(lab_items)}"
        
        prompt = f"""[INST] You are a clinical decision support AI. Analyze this complex patient case:

        Patient Profile:
        - Age: {query.age} years old
        - Current medications: {medications_str}
        - Medical conditions: {conditions_str}
        {lab_context}
        {query.additional_context}
        
        Provide comprehensive analysis:
        1. All potential drug-drug interactions (rank by severity)
        2. Drug-disease interactions
        3. Age-related considerations
        4. Risk stratification (calculate overall risk score)
        5. Recommended monitoring plan with specific timeframes
        6. Safer alternative medications if needed
        7. Priority actions for the healthcare provider
        
        Use evidence-based medicine and be thorough but concise. [/INST]"""
        
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=768)
        
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
        
        start_time = time.time()
        
        with torch.cuda.amp.autocast():
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=400,
                    temperature=0.7,
                    do_sample=True,
                    top_p=0.9,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                    use_cache=True,
                )
        
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        if "[/INST]" in response:
            response = response.split("[/INST]")[-1].strip()
        
        inference_time = time.time() - start_time
        
        # Calculate risk score based on factors
        risk_score = min(10, len(query.medications) * 1.5 + (query.age / 10) - 5)
        risk_level = "Low" if risk_score < 4 else "Moderate" if risk_score < 7 else "High"
        
        return JSONResponse(content={
            "success": True,
            "patient_summary": {
                "medications": query.medications,
                "age": query.age,
                "conditions": query.conditions
            },
            "analysis": response,
            "risk_score": round(risk_score, 1),
            "risk_level": risk_level,
            "inference_time": round(inference_time, 2),
            "complexity_level": "complex"
        })
        
    except Exception as e:
        logger.error(f"Error processing complex interaction: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "model_loaded": model_loaded,
        "model_type": "Fine-tuned Mistral Medical MoE (4-bit Cached)",
        "capabilities": [
            "drug-drug interactions",
            "polypharmacy analysis",
            "risk stratification",
            "clinical decision support"
        ]
    }

@app.get("/api/sample_queries")
async def get_sample_queries():
    """Provide sample queries for demonstration"""
    return {
        "simple_examples": [
            {"drug1": "warfarin", "drug2": "aspirin"},
            {"drug1": "metformin", "drug2": "contrast dye"},
            {"drug1": "simvastatin", "drug2": "clarithromycin"}
        ],
        "complex_examples": [
            {
                "description": "Elderly patient with polypharmacy",
                "medications": ["warfarin", "aspirin", "omeprazole", "metoprolol", "amlodipine"],
                "age": 78,
                "conditions": ["atrial fibrillation", "hypertension", "GERD"],
                "lab_values": {"INR": "3.2", "CrCl": "45 mL/min"}
            },
            {
                "description": "Diabetic patient with cardiovascular disease",
                "medications": ["metformin", "glipizide", "atorvastatin", "lisinopril", "clopidogrel"],
                "age": 65,
                "conditions": ["type 2 diabetes", "CAD s/p stent", "hyperlipidemia"],
                "lab_values": {"HbA1c": "8.2%", "eGFR": "58"}
            }
        ]
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)