"""
Metrics for model evaluation.
Includes accuracy, precision, recall, F1-score, and confusion matrix.
"""

from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    roc_auc_score
)


class MetricTracker:
    """
    Track and average metrics during training/validation.
    """
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        """Reset all tracked metrics."""
        self.metrics = {}
        self.counts = {}
    
    def update(self, metrics_dict: Dict[str, float], batch_size: int = 1):
        """
        Update tracked metrics.
        
        Args:
            metrics_dict: Dictionary of metric names to values
            batch_size: Batch size for weighted average
        """
        for name, value in metrics_dict.items():
            if name not in self.metrics:
                self.metrics[name] = 0.0
                self.counts[name] = 0
            
            self.metrics[name] += value * batch_size
            self.counts[name] += batch_size
    
    def compute(self) -> Dict[str, float]:
        """Compute averaged metrics."""
        return {
            name: self.metrics[name] / self.counts[name]
            for name in self.metrics
        }
    
    def __str__(self) -> str:
        """String representation of current metrics."""
        computed = self.compute()
        return " | ".join(f"{k}: {v:.4f}" for k, v in computed.items())


def compute_accuracy(
    predictions: torch.Tensor,
    targets: torch.Tensor
) -> float:
    """
    Compute accuracy.
    
    Args:
        predictions: Model predictions (logits or probabilities)
        targets: Ground truth labels
        
    Returns:
        Accuracy as a float
    """
    if predictions.dim() > 1:
        predictions = torch.argmax(predictions, dim=1)
    
    correct = (predictions == targets).float().sum()
    total = targets.numel()
    
    return (correct / total).item()


def compute_metrics(
    predictions: np.ndarray,
    targets: np.ndarray,
    class_names: Optional[List[str]] = None,
    average: str = "macro"
) -> Dict[str, float]:
    """
    Compute comprehensive evaluation metrics.
    
    Args:
        predictions: Predicted class indices
        targets: Ground truth class indices
        class_names: Optional list of class names for reporting
        average: Averaging strategy ('micro', 'macro', 'weighted')
        
    Returns:
        Dictionary of metrics
    """
    # Ensure numpy arrays
    if isinstance(predictions, torch.Tensor):
        predictions = predictions.cpu().numpy()
    if isinstance(targets, torch.Tensor):
        targets = targets.cpu().numpy()
    
    # Flatten if needed
    predictions = predictions.flatten()
    targets = targets.flatten()
    
    # Compute metrics
    metrics = {
        "accuracy": accuracy_score(targets, predictions),
        "precision": precision_score(
            targets, predictions, average=average, zero_division=0
        ),
        "recall": recall_score(
            targets, predictions, average=average, zero_division=0
        ),
        "f1_score": f1_score(
            targets, predictions, average=average, zero_division=0
        ),
    }
    
    # Per-class metrics
    if class_names:
        per_class_precision = precision_score(
            targets, predictions, average=None, zero_division=0
        )
        per_class_recall = recall_score(
            targets, predictions, average=None, zero_division=0
        )
        per_class_f1 = f1_score(
            targets, predictions, average=None, zero_division=0
        )
        
        for i, cls in enumerate(class_names):
            if i < len(per_class_precision):
                metrics[f"precision_{cls}"] = per_class_precision[i]
                metrics[f"recall_{cls}"] = per_class_recall[i]
                metrics[f"f1_{cls}"] = per_class_f1[i]
    
    return metrics


def compute_confusion_matrix(
    predictions: np.ndarray,
    targets: np.ndarray,
    normalize: str = None
) -> np.ndarray:
    """
    Compute confusion matrix.
    
    Args:
        predictions: Predicted class indices
        targets: Ground truth class indices
        normalize: 'true', 'pred', 'all' or None
        
    Returns:
        Confusion matrix array
    """
    if isinstance(predictions, torch.Tensor):
        predictions = predictions.cpu().numpy()
    if isinstance(targets, torch.Tensor):
        targets = targets.cpu().numpy()
    
    return confusion_matrix(targets, predictions, normalize=normalize)


def get_classification_report(
    predictions: np.ndarray,
    targets: np.ndarray,
    class_names: Optional[List[str]] = None
) -> str:
    """
    Get detailed classification report as string.
    """
    if isinstance(predictions, torch.Tensor):
        predictions = predictions.cpu().numpy()
    if isinstance(targets, torch.Tensor):
        targets = targets.cpu().numpy()
    
    return classification_report(
        targets, predictions,
        target_names=class_names,
        zero_division=0
    )


def compute_roc_auc(
    probabilities: np.ndarray,
    targets: np.ndarray,
    multi_class: str = "ovr"
) -> float:
    """
    Compute ROC-AUC score for multi-class classification.
    
    Args:
        probabilities: Predicted probabilities of shape (N, C)
        targets: Ground truth labels of shape (N,)
        multi_class: 'ovr' (one-vs-rest) or 'ovo' (one-vs-one)
        
    Returns:
        ROC-AUC score
    """
    if isinstance(probabilities, torch.Tensor):
        probabilities = probabilities.cpu().numpy()
    if isinstance(targets, torch.Tensor):
        targets = targets.cpu().numpy()
    
    try:
        return roc_auc_score(
            targets, probabilities,
            multi_class=multi_class,
            average="macro"
        )
    except ValueError:
        # May fail if some classes are not present in targets
        return 0.0


class AverageMeter:
    """
    Computes and stores the average and current value.
    """
    
    def __init__(self, name: str = "Metric"):
        self.name = name
        self.reset()
    
    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
    
    def update(self, val: float, n: int = 1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
    
    def __str__(self) -> str:
        return f"{self.name}: {self.avg:.4f}"


class TopKAccuracy:
    """
    Compute top-k accuracy.
    """
    
    def __init__(self, k: int = 5):
        self.k = k
    
    def __call__(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor
    ) -> float:
        """
        Compute top-k accuracy.
        
        Args:
            predictions: Model predictions (logits or probabilities)
            targets: Ground truth labels
            
        Returns:
            Top-k accuracy
        """
        with torch.no_grad():
            batch_size = targets.size(0)
            
            _, pred = predictions.topk(self.k, dim=1, largest=True, sorted=True)
            pred = pred.t()
            correct = pred.eq(targets.view(1, -1).expand_as(pred))
            
            correct_k = correct[:self.k].reshape(-1).float().sum(0)
            
            return (correct_k / batch_size).item()


if __name__ == "__main__":
    # Test metrics
    num_samples = 100
    num_classes = 8
    class_names = ["A+", "A-", "B+", "B-", "AB+", "AB-", "O+", "O-"]
    
    # Random predictions and targets
    predictions = np.random.randint(0, num_classes, size=num_samples)
    targets = np.random.randint(0, num_classes, size=num_samples)
    
    # Compute metrics
    metrics = compute_metrics(predictions, targets, class_names)
    print("Metrics:")
    for name, value in metrics.items():
        if not name.startswith(("precision_", "recall_", "f1_")):
            print(f"  {name}: {value:.4f}")
    
    # Confusion Matrix
    cm = compute_confusion_matrix(predictions, targets, normalize="true")
    print(f"\nConfusion Matrix Shape: {cm.shape}")
    
    # Classification Report
    print("\nClassification Report:")
    print(get_classification_report(predictions, targets, class_names))
    
    # Test MetricTracker
    tracker = MetricTracker()
    for i in range(10):
        tracker.update({"loss": np.random.rand(), "accuracy": np.random.rand()}, batch_size=32)
    
    print(f"\nTracked Metrics: {tracker}")
