"""
Custom classification head for the fingerprint blood group detection model.
"""

import torch
import torch.nn as nn
from typing import Optional


class ClassificationHead(nn.Module):
    """
    Custom classification head with dropout and batch normalization.
    
    Architecture:
        Input Features -> Dropout -> FC1 -> ReLU -> BatchNorm -> Dropout -> FC2 -> Output
    """
    
    def __init__(
        self,
        in_features: int,
        num_classes: int,
        hidden_dim: int = 512,
        dropout_rate: float = 0.3,
        use_batch_norm: bool = True
    ):
        """
        Args:
            in_features: Number of input features from backbone
            num_classes: Number of output classes (8 for blood groups)
            hidden_dim: Hidden layer dimension
            dropout_rate: Dropout probability
            use_batch_norm: Whether to use batch normalization
        """
        super().__init__()
        
        self.dropout1 = nn.Dropout(p=dropout_rate)
        
        self.fc1 = nn.Linear(in_features, hidden_dim)
        self.relu = nn.ReLU(inplace=True)
        
        self.bn = nn.BatchNorm1d(hidden_dim) if use_batch_norm else nn.Identity()
        
        self.dropout2 = nn.Dropout(p=dropout_rate + 0.1)  # Slightly higher for final layer
        
        self.fc2 = nn.Linear(hidden_dim, num_classes)
        
        # Initialize weights
        self._init_weights()
        
    def _init_weights(self):
        """Initialize weights using Kaiming/He initialization."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input features of shape (B, in_features)
            
        Returns:
            Logits of shape (B, num_classes)
        """
        x = self.dropout1(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.bn(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        
        return x


class MultiHeadClassificationHead(nn.Module):
    """
    Multi-head classification for additional auxiliary tasks.
    
    Main head: Blood group classification
    Auxiliary heads (optional): Pattern type, quality score, etc.
    """
    
    def __init__(
        self,
        in_features: int,
        num_classes: int = 8,
        hidden_dim: int = 512,
        dropout_rate: float = 0.3,
        auxiliary_tasks: Optional[dict] = None
    ):
        """
        Args:
            in_features: Number of input features
            num_classes: Number of main classification classes
            hidden_dim: Hidden dimension
            dropout_rate: Dropout rate
            auxiliary_tasks: Dict of {task_name: num_classes} for auxiliary tasks
        """
        super().__init__()
        
        # Shared representation
        self.shared = nn.Sequential(
            nn.Dropout(p=dropout_rate),
            nn.Linear(in_features, hidden_dim),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(hidden_dim),
            nn.Dropout(p=dropout_rate)
        )
        
        # Main classification head
        self.main_head = nn.Linear(hidden_dim, num_classes)
        
        # Auxiliary heads
        self.auxiliary_heads = nn.ModuleDict()
        if auxiliary_tasks:
            for task_name, task_classes in auxiliary_tasks.items():
                self.auxiliary_heads[task_name] = nn.Linear(hidden_dim, task_classes)
        
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x: torch.Tensor) -> dict:
        """
        Args:
            x: Input features of shape (B, in_features)
            
        Returns:
            Dict with 'main' and auxiliary task logits
        """
        shared_features = self.shared(x)
        
        outputs = {
            'main': self.main_head(shared_features)
        }
        
        for task_name, head in self.auxiliary_heads.items():
            outputs[task_name] = head(shared_features)
        
        return outputs


if __name__ == "__main__":
    # Test classification heads
    batch_size = 4
    in_features = 1536  # EfficientNet-B3 output features
    num_classes = 8
    
    x = torch.randn(batch_size, in_features)
    
    # Test basic head
    head = ClassificationHead(in_features, num_classes)
    out = head(x)
    print(f"Basic Head: {x.shape} -> {out.shape}")
    
    # Test multi-head
    multi_head = MultiHeadClassificationHead(
        in_features,
        num_classes,
        auxiliary_tasks={'quality': 3}
    )
    outputs = multi_head(x)
    print(f"Multi-Head Main: {outputs['main'].shape}")
    print(f"Multi-Head Quality: {outputs['quality'].shape}")
    
    # Parameter count
    print(f"\nParameters: {sum(p.numel() for p in head.parameters()):,}")
