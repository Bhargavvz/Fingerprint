"""
Explainer service for Grad-CAM visualizations.
"""

import sys
from pathlib import Path
from typing import Dict, Union

import numpy as np
from PIL import Image

# Add project root to path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from src.explainability.gradcam import GradCAM, apply_gradcam
from src.explainability.explanations import generate_explanation, analyze_heatmap

from .model_service import ModelService


class ExplainerService:
    """
    Service for generating model explanations with Grad-CAM.
    """
    
    def __init__(self, model_service: ModelService):
        """
        Initialize explainer service.
        
        Args:
            model_service: Initialized model service
        """
        self.model_service = model_service
        self.gradcam = GradCAM(
            model=model_service.model,
            device=str(model_service.device)
        )
    
    def explain(
        self,
        image: Union[Image.Image, np.ndarray],
        target_class: int = None
    ) -> Dict:
        """
        Generate explanation for a prediction.
        
        Args:
            image: Input image
            target_class: Optional target class for Grad-CAM
            
        Returns:
            Dictionary with prediction, Grad-CAM, and explanation
        """
        # Convert to PIL if needed
        if isinstance(image, np.ndarray):
            pil_image = Image.fromarray(image)
        else:
            pil_image = image
        
        original_array = np.array(pil_image)
        
        # Get prediction first
        prediction = self.model_service.predict(pil_image)
        
        # Apply Grad-CAM
        original, heatmap, overlay, pred_class, confidence = apply_gradcam(
            model=self.model_service.model,
            image=pil_image,
            transform=self.model_service.transform,
            target_class=target_class,
            device=str(self.model_service.device),
            alpha=0.5
        )
        
        # Analyze heatmap
        # Convert heatmap to grayscale for analysis
        if heatmap.ndim == 3:
            heatmap_gray = np.mean(heatmap, axis=2)
        else:
            heatmap_gray = heatmap
        
        heatmap_analysis = analyze_heatmap(heatmap_gray)
        
        # Generate text explanation
        explanation_data = generate_explanation(
            predicted_class=prediction["class_index"],
            confidence=prediction["confidence"],
            probabilities=prediction["probabilities"],
            heatmap=heatmap_gray,
            class_names=self.model_service.CLASSES
        )
        
        return {
            "blood_group": prediction["blood_group"],
            "confidence": prediction["confidence"],
            "probabilities": prediction["probabilities"],
            "original": original,
            "heatmap": heatmap,
            "overlay": overlay,
            "explanation": explanation_data["model_reasoning"],
            "full_explanation": explanation_data,
            "heatmap_analysis": heatmap_analysis
        }
    
    def generate_comparison(
        self,
        image: Union[Image.Image, np.ndarray]
    ) -> Dict:
        """
        Generate Grad-CAM for all classes for comparison.
        
        Args:
            image: Input image
            
        Returns:
            Dictionary with Grad-CAM for each class
        """
        explanations = {}
        
        for i, cls in enumerate(self.model_service.CLASSES):
            result = self.explain(image, target_class=i)
            explanations[cls] = {
                "heatmap": result["heatmap"],
                "overlay": result["overlay"],
                "probability": result["probabilities"][cls]
            }
        
        return explanations


if __name__ == "__main__":
    # Test explainer service
    model_service = ModelService()
    explainer = ExplainerService(model_service)
    
    # Test with random image
    dummy_image = Image.fromarray(np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8))
    
    result = explainer.explain(dummy_image)
    print(f"Blood Group: {result['blood_group']}")
    print(f"Confidence: {result['confidence']:.4f}")
    print(f"Explanation: {result['explanation'][:100]}...")
    print(f"Overlay shape: {result['overlay'].shape}")
