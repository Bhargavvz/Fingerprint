"""
FastAPI Backend for Fingerprint Blood Group Detection.

Run with:
    uvicorn app.main:app --reload --port 8000
"""

import os
import sys
import base64
from io import BytesIO
from pathlib import Path
from datetime import datetime
from typing import Optional

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

import torch
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from PIL import Image
import numpy as np

from .services.model_service import ModelService
from .services.explainer import ExplainerService
from .schemas.prediction import (
    PredictionResponse,
    ExplanationResponse,
    HealthResponse
)


# Initialize FastAPI app
app = FastAPI(
    title="Fingerprint Blood Group Detection API",
    description="""
    ## AI-Powered Blood Group Prediction from Fingerprints
    
    This API provides endpoints for predicting blood groups from fingerprint images
    using a deep learning model with explainable AI capabilities.
    
    ### ⚠️ Disclaimer
    This is an academic research project and should NOT be used for medical diagnosis.
    Blood group determination requires proper laboratory testing.
    
    ### Features
    - Blood group prediction from fingerprint images
    - Grad-CAM visualization for model interpretability
    - Confidence scores and explanations
    """,
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify allowed origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global services (initialized on startup)
model_service: Optional[ModelService] = None
explainer_service: Optional[ExplainerService] = None


@app.on_event("startup")
async def startup_event():
    """Initialize services on startup."""
    global model_service, explainer_service
    
    print("Initializing services...")
    
    # Check for GPU
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # Checkpoint path
    checkpoint_path = os.environ.get(
        "MODEL_CHECKPOINT",
        str(project_root / "checkpoints" / "best_model.pt")
    )
    
    # Initialize services
    try:
        model_service = ModelService(
            checkpoint_path=checkpoint_path if os.path.exists(checkpoint_path) else None,
            device=device
        )
        explainer_service = ExplainerService(model_service)
        print("Services initialized successfully!")
    except Exception as e:
        print(f"Warning: Could not load model checkpoint: {e}")
        print("API will run in demo mode with random predictions.")
        model_service = ModelService(checkpoint_path=None, device=device)
        explainer_service = ExplainerService(model_service)


@app.get("/", response_model=HealthResponse)
async def root():
    """Root endpoint with API status."""
    return {
        "status": "healthy",
        "message": "Fingerprint Blood Group Detection API",
        "version": "1.0.0",
        "timestamp": datetime.now().isoformat()
    }


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "message": "API is running",
        "version": "1.0.0",
        "timestamp": datetime.now().isoformat(),
        "model_loaded": model_service is not None and model_service.is_loaded
    }


@app.post("/api/predict", response_model=PredictionResponse)
async def predict(file: UploadFile = File(...)):
    """
    Predict blood group from fingerprint image.
    
    **Parameters:**
    - file: Fingerprint image file (PNG, JPG, BMP)
    
    **Returns:**
    - blood_group: Predicted blood group (A+, A-, B+, B-, AB+, AB-, O+, O-)
    - confidence: Prediction confidence (0-1)
    - all_probabilities: Probabilities for all blood groups
    - disclaimer: Medical disclaimer
    """
    if model_service is None:
        raise HTTPException(status_code=500, detail="Model service not initialized")
    
    # Validate file type
    allowed_types = ["image/png", "image/jpeg", "image/bmp"]
    if file.content_type not in allowed_types:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid file type. Allowed: {allowed_types}"
        )
    
    try:
        # Read image
        contents = await file.read()
        image = Image.open(BytesIO(contents)).convert("RGB")
        
        # Get prediction
        result = model_service.predict(image)
        
        return PredictionResponse(
            blood_group=result["blood_group"],
            confidence=result["confidence"],
            all_probabilities=result["probabilities"],
            disclaimer="This is an AI prediction for research purposes only. "
                       "Not intended for medical diagnosis."
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/explain", response_model=ExplanationResponse)
async def explain(file: UploadFile = File(...)):
    """
    Get prediction with Grad-CAM explanation.
    
    **Parameters:**
    - file: Fingerprint image file (PNG, JPG, BMP)
    
    **Returns:**
    - blood_group: Predicted blood group
    - confidence: Prediction confidence
    - gradcam_image: Base64 encoded Grad-CAM visualization
    - explanation: Human-readable explanation
    - all_probabilities: Probabilities for all blood groups
    """
    if explainer_service is None:
        raise HTTPException(status_code=500, detail="Explainer service not initialized")
    
    # Validate file type
    allowed_types = ["image/png", "image/jpeg", "image/bmp"]
    if file.content_type not in allowed_types:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid file type. Allowed: {allowed_types}"
        )
    
    try:
        # Read image
        contents = await file.read()
        image = Image.open(BytesIO(contents)).convert("RGB")
        
        # Get explanation
        result = explainer_service.explain(image)
        
        # Encode Grad-CAM image to base64
        gradcam_buffer = BytesIO()
        Image.fromarray(result["overlay"]).save(gradcam_buffer, format="PNG")
        gradcam_base64 = base64.b64encode(gradcam_buffer.getvalue()).decode("utf-8")
        
        return ExplanationResponse(
            blood_group=result["blood_group"],
            confidence=result["confidence"],
            gradcam_image=f"data:image/png;base64,{gradcam_base64}",
            explanation=result["explanation"],
            all_probabilities=result["probabilities"],
            disclaimer="This is an AI prediction for research purposes only. "
                       "Not intended for medical diagnosis."
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/classes")
async def get_classes():
    """Get available blood group classes."""
    return {
        "classes": ["A+", "A-", "B+", "B-", "AB+", "AB-", "O+", "O-"],
        "count": 8
    }


# Error handlers
@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    """Global exception handler."""
    return JSONResponse(
        status_code=500,
        content={
            "detail": "An unexpected error occurred",
            "error": str(exc)
        }
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
