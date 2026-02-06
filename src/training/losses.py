"""
Loss functions for training.
Includes Focal Loss for handling class imbalance and Label Smoothing.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class FocalLoss(nn.Module):
    """
    Focal Loss for handling class imbalance.
    
    Focal Loss reduces the loss contribution from easy examples and focuses
    on hard examples, which is particularly useful for imbalanced datasets.
    
    FL(p_t) = -α_t * (1 - p_t)^γ * log(p_t)
    
    Reference: https://arxiv.org/abs/1708.02002
    """
    
    def __init__(
        self,
        alpha: Optional[torch.Tensor] = None,
        gamma: float = 2.0,
        reduction: str = "mean",
        label_smoothing: float = 0.0
    ):
        """
        Args:
            alpha: Class weights tensor of shape (num_classes,).
                   If None, all classes are weighted equally.
            gamma: Focusing parameter. Higher values focus more on hard examples.
                   gamma=0 is equivalent to cross-entropy.
            reduction: 'none', 'mean' or 'sum'
            label_smoothing: Label smoothing factor
        """
        super().__init__()
        
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.label_smoothing = label_smoothing
        
    def forward(
        self,
        inputs: torch.Tensor,
        targets: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            inputs: Predictions of shape (N, C) where C is number of classes
            targets: Ground truth labels of shape (N,)
            
        Returns:
            Loss value
        """
        # Compute softmax probabilities
        p = F.softmax(inputs, dim=1)
        
        # Get the probability of the correct class
        ce_loss = F.cross_entropy(
            inputs, targets,
            reduction='none',
            label_smoothing=self.label_smoothing
        )
        
        # Get p_t
        p_t = p.gather(1, targets.unsqueeze(1)).squeeze(1)
        
        # Compute focal weight
        focal_weight = (1 - p_t) ** self.gamma
        
        # Apply alpha if provided
        if self.alpha is not None:
            if self.alpha.device != inputs.device:
                self.alpha = self.alpha.to(inputs.device)
            alpha_t = self.alpha.gather(0, targets)
            focal_weight = alpha_t * focal_weight
        
        # Compute focal loss
        loss = focal_weight * ce_loss
        
        # Reduction
        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        else:
            return loss


class LabelSmoothingCrossEntropy(nn.Module):
    """
    Cross-Entropy Loss with Label Smoothing.
    
    Label smoothing prevents the model from becoming overconfident
    by distributing some probability mass to incorrect classes.
    """
    
    def __init__(
        self,
        smoothing: float = 0.1,
        reduction: str = "mean",
        weight: Optional[torch.Tensor] = None
    ):
        """
        Args:
            smoothing: Label smoothing factor (0 = no smoothing)
            reduction: 'none', 'mean' or 'sum'
            weight: Class weights tensor
        """
        super().__init__()
        
        self.smoothing = smoothing
        self.reduction = reduction
        self.weight = weight
        
    def forward(
        self,
        inputs: torch.Tensor,
        targets: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            inputs: Predictions of shape (N, C)
            targets: Ground truth labels of shape (N,)
            
        Returns:
            Loss value
        """
        n_classes = inputs.size(1)
        
        # Create smoothed labels
        with torch.no_grad():
            targets_smooth = torch.zeros_like(inputs)
            targets_smooth.fill_(self.smoothing / (n_classes - 1))
            targets_smooth.scatter_(1, targets.unsqueeze(1), 1 - self.smoothing)
        
        # Compute log softmax
        log_probs = F.log_softmax(inputs, dim=1)
        
        # Compute loss
        loss = -targets_smooth * log_probs
        
        # Apply class weights if provided
        if self.weight is not None:
            if self.weight.device != inputs.device:
                self.weight = self.weight.to(inputs.device)
            loss = loss * self.weight.unsqueeze(0)
        
        loss = loss.sum(dim=1)
        
        # Reduction
        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        else:
            return loss


class CombinedLoss(nn.Module):
    """
    Combine multiple loss functions with weights.
    
    Useful for multi-task learning or when using auxiliary losses.
    """
    
    def __init__(self, losses: dict, weights: Optional[dict] = None):
        """
        Args:
            losses: Dictionary of {name: loss_fn}
            weights: Dictionary of {name: weight}. Default weight is 1.0.
        """
        super().__init__()
        
        self.losses = nn.ModuleDict(losses)
        self.weights = weights or {name: 1.0 for name in losses}
        
    def forward(
        self,
        inputs: torch.Tensor,
        targets: torch.Tensor
    ) -> tuple:
        """
        Compute combined loss.
        
        Returns:
            Tuple of (total_loss, loss_dict)
        """
        loss_dict = {}
        total_loss = 0
        
        for name, loss_fn in self.losses.items():
            loss_val = loss_fn(inputs, targets)
            loss_dict[name] = loss_val
            total_loss = total_loss + self.weights[name] * loss_val
        
        return total_loss, loss_dict


def get_loss_function(
    loss_name: str,
    class_weights: Optional[torch.Tensor] = None,
    **kwargs
) -> nn.Module:
    """
    Factory function to create loss functions.
    
    Args:
        loss_name: Name of loss function
        class_weights: Optional class weights for imbalanced data
        **kwargs: Additional arguments for the loss function
        
    Returns:
        Loss function module
    """
    if loss_name.lower() == "focal":
        return FocalLoss(
            alpha=class_weights,
            gamma=kwargs.get("gamma", 2.0),
            label_smoothing=kwargs.get("label_smoothing", 0.0)
        )
    elif loss_name.lower() == "cross_entropy":
        return nn.CrossEntropyLoss(
            weight=class_weights,
            label_smoothing=kwargs.get("label_smoothing", 0.0)
        )
    elif loss_name.lower() == "label_smoothing":
        return LabelSmoothingCrossEntropy(
            smoothing=kwargs.get("smoothing", 0.1),
            weight=class_weights
        )
    else:
        raise ValueError(f"Unknown loss function: {loss_name}")


if __name__ == "__main__":
    # Test loss functions
    batch_size = 8
    num_classes = 8
    
    # Random predictions and targets
    inputs = torch.randn(batch_size, num_classes)
    targets = torch.randint(0, num_classes, (batch_size,))
    
    # Test Focal Loss
    focal_loss = FocalLoss(gamma=2.0)
    fl = focal_loss(inputs, targets)
    print(f"Focal Loss: {fl.item():.4f}")
    
    # Test Focal Loss with class weights
    weights = torch.ones(num_classes)
    weights[0] = 2.0  # Higher weight for class 0
    focal_loss_weighted = FocalLoss(alpha=weights, gamma=2.0)
    fl_w = focal_loss_weighted(inputs, targets)
    print(f"Weighted Focal Loss: {fl_w.item():.4f}")
    
    # Test Label Smoothing
    label_smooth = LabelSmoothingCrossEntropy(smoothing=0.1)
    ls = label_smooth(inputs, targets)
    print(f"Label Smoothing Loss: {ls.item():.4f}")
    
    # Test standard cross entropy for comparison
    ce = nn.CrossEntropyLoss()(inputs, targets)
    print(f"Cross Entropy Loss: {ce.item():.4f}")
