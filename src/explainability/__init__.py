"""
Explainability module for Fingerprint Blood Group Detection.
Implements Grad-CAM for model interpretability.
"""

from .gradcam import GradCAM, apply_gradcam
from .explanations import generate_explanation, ExplanationGenerator

__all__ = [
    "GradCAM",
    "apply_gradcam",
    "generate_explanation",
    "ExplanationGenerator",
]
