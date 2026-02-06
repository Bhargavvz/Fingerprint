"""
Training module for Fingerprint Blood Group Detection.
Contains trainer, losses, metrics, and callbacks.
"""

from .trainer import Trainer, create_trainer
from .losses import FocalLoss, LabelSmoothingCrossEntropy, get_loss_function
from .metrics import compute_metrics, MetricTracker, get_classification_report
from .callbacks import EarlyStopping, ModelCheckpoint, LRSchedulerCallback

__all__ = [
    "Trainer",
    "create_trainer",
    "FocalLoss",
    "LabelSmoothingCrossEntropy",
    "get_loss_function",
    "compute_metrics",
    "MetricTracker",
    "get_classification_report",
    "EarlyStopping",
    "ModelCheckpoint",
    "LRSchedulerCallback",
]

