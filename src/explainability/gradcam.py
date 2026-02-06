"""
Grad-CAM (Gradient-weighted Class Activation Mapping) implementation.
Visualizes which regions of the fingerprint the model focuses on for predictions.
"""

from typing import List, Optional, Tuple, Union

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image


class GradCAM:
    """
    Gradient-weighted Class Activation Mapping (Grad-CAM).
    
    Produces visual explanations for CNN decisions by using the gradients
    flowing into the final convolutional layer.
    
    Reference: https://arxiv.org/abs/1610.02391
    """
    
    def __init__(
        self,
        model: nn.Module,
        target_layer: Optional[nn.Module] = None,
        device: str = "cpu"
    ):
        """
        Args:
            model: The CNN model to explain
            target_layer: Specific layer to compute Grad-CAM for.
                         If None, will try to find the last conv layer.
            device: Device to run on
        """
        self.model = model
        self.device = device
        self.model.to(device)
        self.model.eval()
        
        # Find target layer if not specified
        if target_layer is None:
            self.target_layer = self._find_target_layer()
        else:
            self.target_layer = target_layer
        
        # Storage for gradients and activations
        self.gradients = None
        self.activations = None
        
        # Register hooks
        self._register_hooks()
    
    def _find_target_layer(self) -> nn.Module:
        """
        Find the last convolutional layer in the model.
        Works with EfficientNet architecture.
        """
        # For EfficientNet from timm
        if hasattr(self.model, 'backbone'):
            backbone = self.model.backbone
            
            # EfficientNet structure: features are in conv_head or similar
            if hasattr(backbone, 'conv_head'):
                return backbone.conv_head
            elif hasattr(backbone, 'blocks'):
                # Get the last block
                return backbone.blocks[-1]
        
        # Fallback: find last Conv2d layer
        target = None
        for module in self.model.modules():
            if isinstance(module, nn.Conv2d):
                target = module
        
        if target is None:
            raise ValueError("Could not find a convolutional layer")
        
        return target
    
    def _register_hooks(self):
        """Register forward and backward hooks for gradient capture."""
        
        def forward_hook(module, input, output):
            self.activations = output.detach()
        
        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0].detach()
        
        self.target_layer.register_forward_hook(forward_hook)
        self.target_layer.register_full_backward_hook(backward_hook)
    
    def __call__(
        self,
        input_tensor: torch.Tensor,
        target_class: Optional[int] = None
    ) -> Tuple[np.ndarray, int, float]:
        """
        Compute Grad-CAM for the input.
        
        Args:
            input_tensor: Input image tensor of shape (1, C, H, W)
            target_class: Class to compute Grad-CAM for. If None, uses predicted class.
            
        Returns:
            Tuple of (heatmap, predicted_class, confidence)
        """
        # Ensure correct shape
        if input_tensor.dim() == 3:
            input_tensor = input_tensor.unsqueeze(0)
        
        input_tensor = input_tensor.to(self.device)
        input_tensor.requires_grad = True
        
        # Forward pass
        output = self.model(input_tensor)
        
        # Get prediction
        probs = F.softmax(output, dim=1)
        pred_class = output.argmax(dim=1).item()
        confidence = probs[0, pred_class].item()
        
        # Use target class or predicted class
        if target_class is None:
            target_class = pred_class
        
        # Zero gradients
        self.model.zero_grad()
        
        # Backward pass for target class
        one_hot = torch.zeros_like(output)
        one_hot[0, target_class] = 1
        output.backward(gradient=one_hot, retain_graph=True)
        
        # Get gradients and activations
        gradients = self.gradients  # (1, C, H', W')
        activations = self.activations  # (1, C, H', W')
        
        # Global average pooling of gradients
        weights = gradients.mean(dim=(2, 3), keepdim=True)  # (1, C, 1, 1)
        
        # Weighted combination of activations
        cam = (weights * activations).sum(dim=1, keepdim=True)  # (1, 1, H', W')
        
        # ReLU and normalize
        cam = F.relu(cam)
        cam = cam.squeeze().cpu().numpy()
        
        # Normalize to [0, 1]
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        
        return cam, pred_class, confidence
    
    def generate_heatmap(
        self,
        cam: np.ndarray,
        target_size: Tuple[int, int],
        colormap: int = cv2.COLORMAP_JET
    ) -> np.ndarray:
        """
        Generate a colored heatmap from CAM.
        
        Args:
            cam: Grad-CAM array
            target_size: (height, width) to resize to
            colormap: OpenCV colormap
            
        Returns:
            Colored heatmap as RGB array
        """
        # Resize CAM to target size
        cam_resized = cv2.resize(cam, (target_size[1], target_size[0]))
        
        # Convert to uint8 and apply colormap
        cam_uint8 = (cam_resized * 255).astype(np.uint8)
        heatmap = cv2.applyColorMap(cam_uint8, colormap)
        
        # Convert BGR to RGB
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
        
        return heatmap
    
    def overlay_heatmap(
        self,
        heatmap: np.ndarray,
        original_image: np.ndarray,
        alpha: float = 0.5
    ) -> np.ndarray:
        """
        Overlay heatmap on original image.
        
        Args:
            heatmap: Colored heatmap (RGB)
            original_image: Original image (RGB)
            alpha: Blending factor
            
        Returns:
            Overlaid image
        """
        # Ensure same size
        if heatmap.shape[:2] != original_image.shape[:2]:
            heatmap = cv2.resize(heatmap, (original_image.shape[1], original_image.shape[0]))
        
        # Ensure same dtype
        if original_image.max() <= 1.0:
            original_image = (original_image * 255).astype(np.uint8)
        
        # Blend
        overlay = cv2.addWeighted(original_image, 1 - alpha, heatmap, alpha, 0)
        
        return overlay


def apply_gradcam(
    model: nn.Module,
    image: Union[np.ndarray, torch.Tensor, Image.Image],
    transform=None,
    target_class: Optional[int] = None,
    device: str = "cpu",
    alpha: float = 0.5
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, int, float]:
    """
    Apply Grad-CAM to an image.
    
    Args:
        model: Trained model
        image: Input image (numpy array, tensor, or PIL Image)
        transform: Transform to apply to image
        target_class: Class to visualize (None = predicted class)
        device: Device to use
        alpha: Overlay blending factor
        
    Returns:
        Tuple of (original, heatmap, overlay, predicted_class, confidence)
    """
    # Convert to numpy if needed
    if isinstance(image, Image.Image):
        original = np.array(image)
    elif isinstance(image, torch.Tensor):
        if image.dim() == 3:
            original = image.permute(1, 2, 0).numpy()
        else:
            original = image.numpy()
        if original.max() <= 1.0:
            original = (original * 255).astype(np.uint8)
    else:
        original = image.copy()
    
    # Apply transform
    if transform is not None:
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image)
        input_tensor = transform(image)
    elif isinstance(image, torch.Tensor):
        input_tensor = image
    else:
        raise ValueError("Transform required for non-tensor input")
    
    # Create Grad-CAM
    gradcam = GradCAM(model, device=device)
    
    # Compute CAM
    cam, pred_class, confidence = gradcam(input_tensor, target_class)
    
    # Generate heatmap
    heatmap = gradcam.generate_heatmap(cam, original.shape[:2])
    
    # Overlay
    overlay = gradcam.overlay_heatmap(heatmap, original, alpha)
    
    return original, heatmap, overlay, pred_class, confidence


class GradCAMPlusPlus(GradCAM):
    """
    Grad-CAM++ with improved localization.
    
    Reference: https://arxiv.org/abs/1710.11063
    """
    
    def __call__(
        self,
        input_tensor: torch.Tensor,
        target_class: Optional[int] = None
    ) -> Tuple[np.ndarray, int, float]:
        """Compute Grad-CAM++ with improved weights."""
        
        if input_tensor.dim() == 3:
            input_tensor = input_tensor.unsqueeze(0)
        
        input_tensor = input_tensor.to(self.device)
        input_tensor.requires_grad = True
        
        # Forward pass
        output = self.model(input_tensor)
        probs = F.softmax(output, dim=1)
        pred_class = output.argmax(dim=1).item()
        confidence = probs[0, pred_class].item()
        
        if target_class is None:
            target_class = pred_class
        
        # Zero gradients
        self.model.zero_grad()
        
        # Backward pass
        one_hot = torch.zeros_like(output)
        one_hot[0, target_class] = 1
        output.backward(gradient=one_hot, retain_graph=True)
        
        # Get gradients and activations
        gradients = self.gradients
        activations = self.activations
        
        # Grad-CAM++ weights
        grad_2 = gradients ** 2
        grad_3 = gradients ** 3
        
        # Sum over spatial dimensions
        sum_activations = activations.sum(dim=(2, 3), keepdim=True)
        
        # Compute alpha
        alpha_num = grad_2
        alpha_denom = 2 * grad_2 + sum_activations * grad_3 + 1e-8
        alpha = alpha_num / alpha_denom
        
        # ReLU on gradients
        weights = (alpha * F.relu(gradients)).sum(dim=(2, 3), keepdim=True)
        
        # Weighted combination
        cam = (weights * activations).sum(dim=1, keepdim=True)
        cam = F.relu(cam)
        cam = cam.squeeze().cpu().numpy()
        
        # Normalize
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        
        return cam, pred_class, confidence


if __name__ == "__main__":
    # Test Grad-CAM
    import sys
    from pathlib import Path
    
    sys.path.insert(0, str(Path(__file__).parent.parent.parent))
    
    from src.models import create_model
    from src.data.augmentation import get_val_transforms
    
    # Create model
    model = create_model("efficientnet_b3_cbam", num_classes=8, pretrained=True)
    model.eval()
    
    # Create random input
    x = torch.randn(1, 3, 224, 224)
    
    # Apply Grad-CAM
    gradcam = GradCAM(model)
    cam, pred_class, confidence = gradcam(x)
    
    print(f"CAM shape: {cam.shape}")
    print(f"Predicted class: {pred_class}")
    print(f"Confidence: {confidence:.4f}")
    
    # Generate heatmap
    heatmap = gradcam.generate_heatmap(cam, (224, 224))
    print(f"Heatmap shape: {heatmap.shape}")
