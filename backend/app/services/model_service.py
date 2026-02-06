"""
Model inference service.
Handles model loading and prediction.
"""

import os
import sys
from pathlib import Path
from typing import Dict, Optional, Union

import numpy as np
import torch
from PIL import Image

# Add project root to path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from src.models import create_model
from src.data.augmentation import get_val_transforms
from src.data.preprocessing import preprocess_image


class ModelService:
    """
    Service for model inference.
    """
    
    CLASSES = ["A+", "A-", "B+", "B-", "AB+", "AB-", "O+", "O-"]
    
    def __init__(
        self,
        checkpoint_path: Optional[str] = None,
        device: str = "cpu",
        model_name: str = "efficientnet_b3_cbam"
    ):
        """
        Initialize model service.
        
        Args:
            checkpoint_path: Path to model checkpoint
            device: Device to run on
            model_name: Model architecture name
        """
        self.device = torch.device(device)
        self.model_name = model_name
        self.checkpoint_path = checkpoint_path
        self.is_loaded = False
        
        # Create model
        self.model = create_model(
            model_name=model_name,
            num_classes=len(self.CLASSES),
            pretrained=True
        )
        self.model.to(self.device)
        
        # Load checkpoint if available
        if checkpoint_path and os.path.exists(checkpoint_path):
            self._load_checkpoint(checkpoint_path)
            self.is_loaded = True
        else:
            print("No checkpoint provided. Model initialized with pretrained weights.")
        
        self.model.eval()
        
        # Transform
        self.transform = get_val_transforms(image_size=224)
    
    def _load_checkpoint(self, checkpoint_path: str):
        """Load model weights from checkpoint."""
        print(f"Loading checkpoint: {checkpoint_path}")
        
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        if "model_state_dict" in checkpoint:
            self.model.load_state_dict(checkpoint["model_state_dict"])
        else:
            self.model.load_state_dict(checkpoint)
        
        print("Checkpoint loaded successfully!")
    
    def predict(
        self,
        image: Union[Image.Image, np.ndarray, str]
    ) -> Dict:
        """
        Make a prediction on an image.
        
        Args:
            image: Input image (PIL Image, numpy array, or path)
            
        Returns:
            Dictionary with prediction results
        """
        # Load image if path
        if isinstance(image, str):
            image = Image.open(image).convert("RGB")
        elif isinstance(image, np.ndarray):
            image = Image.fromarray(image)
        
        # Apply transform
        input_tensor = self.transform(image)
        input_tensor = input_tensor.unsqueeze(0).to(self.device)
        
        # Inference
        with torch.no_grad():
            outputs = self.model(input_tensor)
            probabilities = torch.softmax(outputs, dim=1)[0]
            predicted_class = torch.argmax(probabilities).item()
            confidence = probabilities[predicted_class].item()
        
        # Create probability dictionary
        prob_dict = {
            cls: prob.item()
            for cls, prob in zip(self.CLASSES, probabilities)
        }
        
        return {
            "blood_group": self.CLASSES[predicted_class],
            "class_index": predicted_class,
            "confidence": confidence,
            "probabilities": prob_dict
        }
    
    def predict_batch(
        self,
        images: list
    ) -> list:
        """
        Make predictions on a batch of images.
        
        Args:
            images: List of images
            
        Returns:
            List of prediction results
        """
        return [self.predict(img) for img in images]
    
    def get_features(
        self,
        image: Union[Image.Image, np.ndarray]
    ) -> np.ndarray:
        """
        Extract features from an image.
        
        Args:
            image: Input image
            
        Returns:
            Feature vector
        """
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image)
        
        input_tensor = self.transform(image)
        input_tensor = input_tensor.unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            features = self.model.get_features(input_tensor)
        
        return features.cpu().numpy()


if __name__ == "__main__":
    # Test model service
    service = ModelService()
    
    # Test with random image
    dummy_image = Image.fromarray(np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8))
    
    result = service.predict(dummy_image)
    print(f"Prediction: {result['blood_group']}")
    print(f"Confidence: {result['confidence']:.4f}")
    print(f"Probabilities: {result['probabilities']}")
