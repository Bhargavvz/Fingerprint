"""
Training callbacks for model training.
Includes early stopping, model checkpointing, and learning rate scheduling.
"""

import os
from pathlib import Path
from typing import Any, Dict, Optional

import torch
import torch.nn as nn


class EarlyStopping:
    """
    Early stopping to stop training when a monitored metric stops improving.
    """
    
    def __init__(
        self,
        patience: int = 10,
        min_delta: float = 0.0,
        mode: str = "max",
        verbose: bool = True
    ):
        """
        Args:
            patience: Number of epochs to wait for improvement
            min_delta: Minimum change to qualify as improvement
            mode: 'min' or 'max' depending on metric
            verbose: Whether to print messages
        """
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.verbose = verbose
        
        self.counter = 0
        self.best_score = None
        self.should_stop = False
        self.best_epoch = 0
        
        if mode == "min":
            self.is_better = lambda score, best: score < best - min_delta
        else:
            self.is_better = lambda score, best: score > best + min_delta
    
    def __call__(self, score: float, epoch: int) -> bool:
        """
        Check if training should stop.
        
        Args:
            score: Current metric value
            epoch: Current epoch number
            
        Returns:
            True if training should stop
        """
        if self.best_score is None:
            self.best_score = score
            self.best_epoch = epoch
            return False
        
        if self.is_better(score, self.best_score):
            self.best_score = score
            self.best_epoch = epoch
            self.counter = 0
            return False
        else:
            self.counter += 1
            if self.verbose:
                print(f"  EarlyStopping: {self.counter}/{self.patience}")
            
            if self.counter >= self.patience:
                self.should_stop = True
                if self.verbose:
                    print(f"  Early stopping triggered. Best epoch: {self.best_epoch}")
                return True
        
        return False
    
    def reset(self):
        """Reset early stopping state."""
        self.counter = 0
        self.best_score = None
        self.should_stop = False
        self.best_epoch = 0


class ModelCheckpoint:
    """
    Save model checkpoints based on monitored metric.
    """
    
    def __init__(
        self,
        dirpath: str = "checkpoints",
        filename: str = "model_{epoch:02d}_{val_accuracy:.4f}",
        monitor: str = "val_accuracy",
        mode: str = "max",
        save_best_only: bool = True,
        save_last: bool = True,
        verbose: bool = True
    ):
        """
        Args:
            dirpath: Directory to save checkpoints
            filename: Checkpoint filename template
            monitor: Metric to monitor
            mode: 'min' or 'max'
            save_best_only: Only save the best model
            save_last: Always save the last model
            verbose: Print save messages
        """
        self.dirpath = Path(dirpath)
        self.filename = filename
        self.monitor = monitor
        self.mode = mode
        self.save_best_only = save_best_only
        self.save_last = save_last
        self.verbose = verbose
        
        self.best_score = None
        self.best_path = None
        
        # Create directory
        self.dirpath.mkdir(parents=True, exist_ok=True)
        
        if mode == "min":
            self.is_better = lambda score, best: score < best
        else:
            self.is_better = lambda score, best: score > best
    
    def __call__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        epoch: int,
        metrics: Dict[str, float],
        scheduler: Optional[Any] = None
    ) -> Optional[str]:
        """
        Save checkpoint if criteria met.
        
        Returns:
            Path to saved checkpoint or None
        """
        score = metrics.get(self.monitor, 0.0)
        
        # Prepare checkpoint
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "metrics": metrics,
            "best_score": self.best_score or score,
        }
        
        if scheduler is not None:
            checkpoint["scheduler_state_dict"] = scheduler.state_dict()
        
        saved_path = None
        
        # Check if this is the best model
        if self.best_score is None or self.is_better(score, self.best_score):
            self.best_score = score
            
            # Format filename
            format_dict = {"epoch": epoch, **metrics}
            formatted_name = self.filename.format(**format_dict)
            
            # Save best model
            best_path = self.dirpath / "best.pt"
            torch.save(checkpoint, best_path)
            self.best_path = str(best_path)
            
            if self.verbose:
                print(f"  Saved best model: {self.monitor}={score:.4f}")
            
            saved_path = str(best_path)
        elif not self.save_best_only:
            # Save regular checkpoint
            format_dict = {"epoch": epoch, **metrics}
            formatted_name = self.filename.format(**format_dict)
            checkpoint_path = self.dirpath / f"{formatted_name}.pt"
            torch.save(checkpoint, checkpoint_path)
            saved_path = str(checkpoint_path)
        
        # Always save last model
        if self.save_last:
            last_path = self.dirpath / "last.pt"
            torch.save(checkpoint, last_path)
        
        return saved_path
    
    @staticmethod
    def load_checkpoint(
        path: str,
        model: nn.Module,
        optimizer: Optional[torch.optim.Optimizer] = None,
        scheduler: Optional[Any] = None,
        device: str = "cpu"
    ) -> Dict[str, Any]:
        """
        Load a checkpoint.
        
        Returns:
            Dictionary with checkpoint info
        """
        checkpoint = torch.load(path, map_location=device)
        
        model.load_state_dict(checkpoint["model_state_dict"])
        
        if optimizer is not None and "optimizer_state_dict" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        
        if scheduler is not None and "scheduler_state_dict" in checkpoint:
            scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        
        return {
            "epoch": checkpoint.get("epoch", 0),
            "metrics": checkpoint.get("metrics", {}),
            "best_score": checkpoint.get("best_score", None),
        }


class LRSchedulerCallback:
    """
    Learning rate scheduler callback with warmup support.
    """
    
    def __init__(
        self,
        scheduler: Any,
        warmup_epochs: int = 0,
        warmup_start_lr: float = 1e-6,
        verbose: bool = True
    ):
        """
        Args:
            scheduler: PyTorch learning rate scheduler
            warmup_epochs: Number of warmup epochs
            warmup_start_lr: Starting learning rate for warmup
            verbose: Print LR changes
        """
        self.scheduler = scheduler
        self.warmup_epochs = warmup_epochs
        self.warmup_start_lr = warmup_start_lr
        self.verbose = verbose
        
        self.base_lrs = None
    
    def step(
        self,
        epoch: int,
        optimizer: torch.optim.Optimizer,
        metric: Optional[float] = None
    ):
        """
        Step the scheduler.
        
        Args:
            epoch: Current epoch
            optimizer: Optimizer for LR warmup
            metric: Metric value for ReduceLROnPlateau
        """
        # Warmup
        if epoch < self.warmup_epochs:
            if self.base_lrs is None:
                self.base_lrs = [pg["lr"] for pg in optimizer.param_groups]
            
            # Linear warmup
            warmup_factor = (epoch + 1) / self.warmup_epochs
            for pg, base_lr in zip(optimizer.param_groups, self.base_lrs):
                pg["lr"] = self.warmup_start_lr + (base_lr - self.warmup_start_lr) * warmup_factor
            
            if self.verbose:
                current_lr = optimizer.param_groups[0]["lr"]
                print(f"  Warmup LR: {current_lr:.6f}")
        else:
            # Regular scheduler step
            if hasattr(self.scheduler, "step"):
                if metric is not None and hasattr(self.scheduler, "is_better"):
                    self.scheduler.step(metric)
                else:
                    self.scheduler.step()
            
            if self.verbose:
                current_lr = optimizer.param_groups[0]["lr"]
                print(f"  LR: {current_lr:.6f}")
    
    def get_lr(self, optimizer: torch.optim.Optimizer) -> float:
        """Get current learning rate."""
        return optimizer.param_groups[0]["lr"]


class GradientClipping:
    """
    Gradient clipping callback.
    """
    
    def __init__(
        self,
        max_norm: float = 1.0,
        norm_type: float = 2.0
    ):
        self.max_norm = max_norm
        self.norm_type = norm_type
    
    def __call__(self, model: nn.Module) -> float:
        """
        Clip gradients and return the total norm.
        """
        return torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            self.max_norm,
            self.norm_type
        )


if __name__ == "__main__":
    # Test callbacks
    
    # Test Early Stopping
    print("Testing Early Stopping:")
    early_stop = EarlyStopping(patience=3, mode="max")
    
    scores = [0.70, 0.75, 0.80, 0.79, 0.78, 0.77, 0.76]
    for epoch, score in enumerate(scores):
        should_stop = early_stop(score, epoch)
        print(f"  Epoch {epoch}: score={score:.2f}, stop={should_stop}")
        if should_stop:
            break
    
    # Test Model Checkpoint
    print("\nTesting Model Checkpoint:")
    
    class DummyModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc = nn.Linear(10, 8)
    
    model = DummyModel()
    optimizer = torch.optim.Adam(model.parameters())
    
    checkpoint = ModelCheckpoint(
        dirpath="test_checkpoints",
        monitor="val_accuracy",
        mode="max"
    )
    
    # Simulate training
    metrics = {"val_accuracy": 0.85, "val_loss": 0.5}
    saved = checkpoint(model, optimizer, epoch=0, metrics=metrics)
    print(f"  Saved: {saved}")
    
    # Cleanup
    import shutil
    if os.path.exists("test_checkpoints"):
        shutil.rmtree("test_checkpoints")
