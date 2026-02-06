"""
EfficientNet-B3 with CBAM attention mechanism for blood group classification.
This is the main hybrid model combining pretrained EfficientNet with attention.
"""

from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import timm

from .attention import CBAM, SEBlock, ECABlock
from .classifier import ClassificationHead


class EfficientNetCBAM(nn.Module):
    """
    Hybrid EfficientNet-B3 model with CBAM attention.
    
    The model combines:
    1. EfficientNet-B3 pretrained on ImageNet as the backbone
    2. CBAM attention module after feature extraction
    3. Custom classification head for blood group prediction
    
    Architecture:
        Input (224x224x3) 
        -> EfficientNet Backbone 
        -> CBAM Attention 
        -> Global Average Pooling 
        -> Classification Head 
        -> 8 Blood Group Classes
    """
    
    # Blood group classes
    CLASSES = ["A+", "A-", "B+", "B-", "AB+", "AB-", "O+", "O-"]
    
    def __init__(
        self,
        num_classes: int = 8,
        pretrained: bool = True,
        backbone: str = "efficientnet_b3",
        attention_type: str = "cbam",
        attention_reduction: int = 16,
        dropout: float = 0.3,
        freeze_backbone: bool = False,
        freeze_layers: int = 0
    ):
        """
        Args:
            num_classes: Number of output classes (8 for blood groups)
            pretrained: Whether to use ImageNet pretrained weights
            backbone: Backbone architecture from timm
            attention_type: Type of attention ('cbam', 'se', 'eca', 'none')
            attention_reduction: Reduction ratio for attention modules
            dropout: Dropout rate for classification head
            freeze_backbone: Whether to freeze all backbone layers
            freeze_layers: Number of initial layers to freeze (if not freezing all)
        """
        super().__init__()
        
        self.num_classes = num_classes
        self.backbone_name = backbone
        self.attention_type = attention_type
        
        # Create backbone (without classifier)
        self.backbone = timm.create_model(
            backbone,
            pretrained=pretrained,
            num_classes=0,  # Remove classifier
            global_pool=""   # Remove global pooling
        )
        
        # Get feature dimension from backbone
        self.feature_dim = self._get_feature_dim()
        
        # Attention module
        self.attention = self._create_attention(attention_type, attention_reduction)
        
        # Global average pooling
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        
        # Classification head
        self.classifier = ClassificationHead(
            in_features=self.feature_dim,
            num_classes=num_classes,
            hidden_dim=512,
            dropout_rate=dropout
        )
        
        # Freeze layers if specified
        if freeze_backbone:
            self._freeze_backbone()
        elif freeze_layers > 0:
            self._freeze_initial_layers(freeze_layers)
    
    def _get_feature_dim(self) -> int:
        """Get the feature dimension from the backbone."""
        # EfficientNet-B3 has 1536 features
        backbone_dims = {
            "efficientnet_b0": 1280,
            "efficientnet_b1": 1280,
            "efficientnet_b2": 1408,
            "efficientnet_b3": 1536,
            "efficientnet_b4": 1792,
            "efficientnet_b5": 2048,
            "resnet50": 2048,
            "resnet101": 2048,
        }
        return backbone_dims.get(self.backbone_name, 1536)
    
    def _create_attention(
        self,
        attention_type: str,
        reduction: int
    ) -> nn.Module:
        """Create attention module based on type."""
        if attention_type.lower() == "cbam":
            return CBAM(self.feature_dim, reduction)
        elif attention_type.lower() == "se":
            return SEBlock(self.feature_dim, reduction)
        elif attention_type.lower() == "eca":
            return ECABlock(self.feature_dim)
        elif attention_type.lower() == "none":
            return nn.Identity()
        else:
            raise ValueError(f"Unknown attention type: {attention_type}")
    
    def _freeze_backbone(self):
        """Freeze all backbone parameters."""
        for param in self.backbone.parameters():
            param.requires_grad = False
        print(f"Frozen all backbone parameters")
    
    def _freeze_initial_layers(self, num_layers: int):
        """Freeze initial layers of the backbone."""
        # For EfficientNet, freeze the stem and first N blocks
        frozen_count = 0
        
        for name, param in self.backbone.named_parameters():
            if frozen_count < num_layers or "stem" in name:
                param.requires_grad = False
                frozen_count += 1
        
        print(f"Frozen {frozen_count} initial parameters")
    
    def unfreeze_backbone(self):
        """Unfreeze all backbone parameters for fine-tuning."""
        for param in self.backbone.parameters():
            param.requires_grad = True
        print("Unfrozen all backbone parameters")
    
    def get_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extract features before the classifier.
        
        Args:
            x: Input tensor of shape (B, 3, H, W)
            
        Returns:
            Feature tensor of shape (B, feature_dim)
        """
        # Backbone features
        features = self.backbone(x)  # (B, C, H, W)
        
        # Attention
        features = self.attention(features)  # (B, C, H, W)
        
        # Global pooling
        features = self.global_pool(features)  # (B, C, 1, 1)
        features = features.flatten(1)  # (B, C)
        
        return features
    
    def forward(
        self,
        x: torch.Tensor,
        return_features: bool = False
    ) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input tensor of shape (B, 3, H, W)
            return_features: If True, also return intermediate features
            
        Returns:
            Logits of shape (B, num_classes), optionally with features
        """
        # Extract features
        features = self.get_features(x)
        
        # Classification
        logits = self.classifier(features)
        
        if return_features:
            return logits, features
        
        return logits
    
    def predict(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Make predictions with probabilities.
        
        Args:
            x: Input tensor of shape (B, 3, H, W)
            
        Returns:
            Tuple of (predicted_classes, probabilities)
        """
        self.eval()
        with torch.no_grad():
            logits = self.forward(x)
            probabilities = torch.softmax(logits, dim=1)
            predicted_classes = torch.argmax(probabilities, dim=1)
        
        return predicted_classes, probabilities
    
    def predict_blood_group(self, x: torch.Tensor) -> Dict[str, float]:
        """
        Predict blood group with class names.
        
        Args:
            x: Single image tensor of shape (1, 3, H, W)
            
        Returns:
            Dictionary mapping blood group names to probabilities
        """
        _, probabilities = self.predict(x)
        probs = probabilities[0].cpu().numpy()
        
        return {
            cls: float(prob)
            for cls, prob in zip(self.CLASSES, probs)
        }
    
    def get_attention_weights(
        self,
        x: torch.Tensor
    ) -> Optional[torch.Tensor]:
        """
        Get attention weights for visualization.
        
        Only works if attention_type is 'cbam' or 'se'.
        """
        if self.attention_type == "none":
            return None
        
        # Get features before attention
        with torch.no_grad():
            features = self.backbone(x)
            
            if isinstance(self.attention, CBAM):
                # Get channel and spatial attention
                channel_att = self.attention.channel_attention(features)
                refined = features * channel_att
                spatial_att = self.attention.spatial_attention(refined)
                return spatial_att
            else:
                return None
    
    @property
    def device(self) -> torch.device:
        """Get the device the model is on."""
        return next(self.parameters()).device


def create_model(
    model_name: str = "efficientnet_b3_cbam",
    num_classes: int = 8,
    pretrained: bool = True,
    **kwargs
) -> nn.Module:
    """
    Factory function to create models.
    
    Args:
        model_name: Model variant name
        num_classes: Number of output classes
        pretrained: Whether to use pretrained weights
        **kwargs: Additional model arguments
        
    Returns:
        Created model
    """
    model_configs = {
        "efficientnet_b3_cbam": {
            "backbone": "efficientnet_b3",
            "attention_type": "cbam"
        },
        "efficientnet_b3_se": {
            "backbone": "efficientnet_b3",
            "attention_type": "se"
        },
        "efficientnet_b3": {
            "backbone": "efficientnet_b3",
            "attention_type": "none"
        },
        "efficientnet_b0_cbam": {
            "backbone": "efficientnet_b0",
            "attention_type": "cbam"
        },
        "resnet50_cbam": {
            "backbone": "resnet50",
            "attention_type": "cbam"
        },
    }
    
    if model_name not in model_configs:
        raise ValueError(f"Unknown model: {model_name}. Available: {list(model_configs.keys())}")
    
    config = model_configs[model_name]
    config.update(kwargs)
    
    return EfficientNetCBAM(
        num_classes=num_classes,
        pretrained=pretrained,
        **config
    )


def count_parameters(model: nn.Module) -> Dict[str, int]:
    """Count trainable and total parameters."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    return {
        "total": total,
        "trainable": trainable,
        "frozen": total - trainable
    }


if __name__ == "__main__":
    # Test model creation
    print("Creating EfficientNet-B3 + CBAM model...")
    
    model = create_model(
        "efficientnet_b3_cbam",
        num_classes=8,
        pretrained=True,
        dropout=0.3
    )
    
    # Print model summary
    params = count_parameters(model)
    print(f"\nModel Parameters:")
    print(f"  Total: {params['total']:,}")
    print(f"  Trainable: {params['trainable']:,}")
    print(f"  Frozen: {params['frozen']:,}")
    
    # Test forward pass
    batch_size = 4
    x = torch.randn(batch_size, 3, 224, 224)
    
    print(f"\nInput shape: {x.shape}")
    
    # Forward pass
    logits = model(x)
    print(f"Output shape: {logits.shape}")
    
    # With features
    logits, features = model(x, return_features=True)
    print(f"Features shape: {features.shape}")
    
    # Prediction
    classes, probs = model.predict(x)
    print(f"Predicted classes: {classes}")
    print(f"Probabilities shape: {probs.shape}")
    
    # Blood group prediction
    blood_groups = model.predict_blood_group(x[:1])
    print(f"\nBlood group probabilities:")
    for bg, prob in sorted(blood_groups.items(), key=lambda x: -x[1]):
        print(f"  {bg}: {prob:.4f}")
