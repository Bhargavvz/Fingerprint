"""
Pydantic schemas for API requests and responses.
"""

from typing import Dict, Optional
from pydantic import BaseModel, Field


class PredictionResponse(BaseModel):
    """Response schema for prediction endpoint."""
    
    blood_group: str = Field(
        ...,
        description="Predicted blood group",
        example="A+"
    )
    confidence: float = Field(
        ...,
        description="Prediction confidence (0-1)",
        ge=0,
        le=1,
        example=0.95
    )
    all_probabilities: Dict[str, float] = Field(
        ...,
        description="Probabilities for all blood groups",
        example={
            "A+": 0.95, "A-": 0.02, "B+": 0.01, "B-": 0.005,
            "AB+": 0.005, "AB-": 0.005, "O+": 0.003, "O-": 0.002
        }
    )
    disclaimer: str = Field(
        ...,
        description="Medical disclaimer",
        example="This is an AI prediction for research purposes only."
    )
    
    class Config:
        json_schema_extra = {
            "example": {
                "blood_group": "A+",
                "confidence": 0.95,
                "all_probabilities": {
                    "A+": 0.95, "A-": 0.02, "B+": 0.01, "B-": 0.005,
                    "AB+": 0.005, "AB-": 0.005, "O+": 0.003, "O-": 0.002
                },
                "disclaimer": "This is an AI prediction for research purposes only."
            }
        }


class ExplanationResponse(BaseModel):
    """Response schema for explanation endpoint."""
    
    blood_group: str = Field(..., description="Predicted blood group")
    confidence: float = Field(..., description="Prediction confidence", ge=0, le=1)
    gradcam_image: str = Field(
        ...,
        description="Base64 encoded Grad-CAM visualization"
    )
    explanation: str = Field(
        ...,
        description="Human-readable explanation of the prediction"
    )
    all_probabilities: Dict[str, float] = Field(
        ...,
        description="Probabilities for all blood groups"
    )
    disclaimer: str = Field(..., description="Medical disclaimer")
    
    class Config:
        json_schema_extra = {
            "example": {
                "blood_group": "A+",
                "confidence": 0.92,
                "gradcam_image": "data:image/png;base64,iVBORw0KGgo...",
                "explanation": "The model analyzed the fingerprint ridge patterns...",
                "all_probabilities": {
                    "A+": 0.92, "A-": 0.03, "B+": 0.02, "B-": 0.01,
                    "AB+": 0.01, "AB-": 0.005, "O+": 0.003, "O-": 0.002
                },
                "disclaimer": "This is an AI prediction for research purposes only."
            }
        }


class HealthResponse(BaseModel):
    """Response schema for health check endpoint."""
    
    status: str = Field(..., description="API status", example="healthy")
    message: str = Field(..., description="Status message")
    version: str = Field(..., description="API version", example="1.0.0")
    timestamp: str = Field(..., description="Current timestamp")
    model_loaded: Optional[bool] = Field(
        None,
        description="Whether the model is loaded"
    )


class ErrorResponse(BaseModel):
    """Response schema for errors."""
    
    detail: str = Field(..., description="Error message")
    error: Optional[str] = Field(None, description="Detailed error")
