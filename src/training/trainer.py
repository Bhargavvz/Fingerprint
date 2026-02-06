"""
Main trainer class for model training.
Supports mixed precision training, gradient accumulation, and comprehensive logging.
"""

import os
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from .callbacks import EarlyStopping, ModelCheckpoint, LRSchedulerCallback
from .losses import FocalLoss, get_loss_function
from .metrics import compute_metrics, compute_confusion_matrix, MetricTracker, AverageMeter


class Trainer:
    """
    Complete training pipeline with mixed precision, callbacks, and logging.
    """
    
    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        criterion: nn.Module,
        device: torch.device,
        scheduler: Optional[Any] = None,
        config: Optional[Dict] = None
    ):
        """
        Args:
            model: PyTorch model to train
            optimizer: Optimizer instance
            criterion: Loss function
            device: Device to train on
            scheduler: Optional learning rate scheduler
            config: Training configuration dictionary
        """
        self.model = model.to(device)
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device
        self.scheduler = scheduler
        self.config = config or {}
        
        # Mixed precision
        self.use_amp = self.config.get("mixed_precision", True) and device.type == "cuda"
        self.scaler = GradScaler() if self.use_amp else None
        
        # Gradient accumulation
        self.grad_accum_steps = self.config.get("gradient_accumulation_steps", 1)
        
        # Callbacks
        self.early_stopping = EarlyStopping(
            patience=self.config.get("early_stopping_patience", 10),
            mode="max",
            verbose=True
        )
        
        self.checkpoint = ModelCheckpoint(
            dirpath=self.config.get("checkpoint_dir", "checkpoints"),
            monitor="val_accuracy",
            mode="max",
            save_best_only=self.config.get("save_best_only", True),
            save_last=True,
            verbose=True
        )
        
        if scheduler:
            self.lr_callback = LRSchedulerCallback(
                scheduler=scheduler,
                warmup_epochs=self.config.get("warmup_epochs", 0),
                verbose=True
            )
        else:
            self.lr_callback = None
        
        # Logging
        log_dir = self.config.get("log_dir", "outputs/logs")
        self.writer = SummaryWriter(log_dir=log_dir) if self.config.get("tensorboard", True) else None
        
        # Class names for metrics
        self.class_names = self.config.get("class_names", [
            "A+", "A-", "B+", "B-", "AB+", "AB-", "O+", "O-"
        ])
        
        # History
        self.history = {
            "train_loss": [],
            "train_accuracy": [],
            "val_loss": [],
            "val_accuracy": [],
            "learning_rate": []
        }
    
    def train_epoch(
        self,
        train_loader: DataLoader,
        epoch: int
    ) -> Dict[str, float]:
        """
        Train for one epoch.
        
        Returns:
            Dictionary of training metrics
        """
        self.model.train()
        
        loss_meter = AverageMeter("Loss")
        acc_meter = AverageMeter("Accuracy")
        
        all_preds = []
        all_targets = []
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch} [Train]")
        
        self.optimizer.zero_grad()
        
        for batch_idx, (images, targets) in enumerate(pbar):
            images = images.to(self.device)
            targets = targets.to(self.device)
            
            # Forward pass with mixed precision
            if self.use_amp:
                with autocast():
                    outputs = self.model(images)
                    loss = self.criterion(outputs, targets)
                    loss = loss / self.grad_accum_steps
                
                # Backward pass
                self.scaler.scale(loss).backward()
                
                # Gradient accumulation
                if (batch_idx + 1) % self.grad_accum_steps == 0:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                    self.optimizer.zero_grad()
            else:
                outputs = self.model(images)
                loss = self.criterion(outputs, targets)
                loss = loss / self.grad_accum_steps
                
                loss.backward()
                
                if (batch_idx + 1) % self.grad_accum_steps == 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                    self.optimizer.step()
                    self.optimizer.zero_grad()
            
            # Compute accuracy
            _, preds = torch.max(outputs, 1)
            acc = (preds == targets).float().mean()
            
            # Update meters
            batch_size = images.size(0)
            loss_meter.update(loss.item() * self.grad_accum_steps, batch_size)
            acc_meter.update(acc.item(), batch_size)
            
            # Store predictions
            all_preds.append(preds.cpu())
            all_targets.append(targets.cpu())
            
            # Update progress bar
            pbar.set_postfix({
                "loss": f"{loss_meter.avg:.4f}",
                "acc": f"{acc_meter.avg:.4f}"
            })
        
        # Compute epoch metrics
        all_preds = torch.cat(all_preds).numpy()
        all_targets = torch.cat(all_targets).numpy()
        
        metrics = compute_metrics(all_preds, all_targets, self.class_names)
        metrics["loss"] = loss_meter.avg
        
        return metrics
    
    @torch.no_grad()
    def validate(
        self,
        val_loader: DataLoader,
        epoch: int
    ) -> Dict[str, float]:
        """
        Validate the model.
        
        Returns:
            Dictionary of validation metrics
        """
        self.model.eval()
        
        loss_meter = AverageMeter("Loss")
        
        all_preds = []
        all_targets = []
        all_probs = []
        
        pbar = tqdm(val_loader, desc=f"Epoch {epoch} [Val]")
        
        for images, targets in pbar:
            images = images.to(self.device)
            targets = targets.to(self.device)
            
            if self.use_amp:
                with autocast():
                    outputs = self.model(images)
                    loss = self.criterion(outputs, targets)
            else:
                outputs = self.model(images)
                loss = self.criterion(outputs, targets)
            
            # Get predictions and probabilities
            probs = torch.softmax(outputs, dim=1)
            _, preds = torch.max(outputs, 1)
            
            # Update meters
            batch_size = images.size(0)
            loss_meter.update(loss.item(), batch_size)
            
            # Store results
            all_preds.append(preds.cpu())
            all_targets.append(targets.cpu())
            all_probs.append(probs.cpu())
            
            pbar.set_postfix({"loss": f"{loss_meter.avg:.4f}"})
        
        # Compute metrics
        all_preds = torch.cat(all_preds).numpy()
        all_targets = torch.cat(all_targets).numpy()
        all_probs = torch.cat(all_probs).numpy()
        
        metrics = compute_metrics(all_preds, all_targets, self.class_names)
        metrics["loss"] = loss_meter.avg
        
        # Compute confusion matrix
        cm = compute_confusion_matrix(all_preds, all_targets)
        
        return metrics, cm
    
    def fit(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        epochs: int,
        start_epoch: int = 0
    ) -> Dict[str, list]:
        """
        Full training loop.
        
        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
            epochs: Total number of epochs
            start_epoch: Starting epoch (for resuming)
            
        Returns:
            Training history dictionary
        """
        print(f"\nStarting training on {self.device}")
        print(f"Mixed precision: {self.use_amp}")
        print(f"Gradient accumulation steps: {self.grad_accum_steps}")
        print(f"Total epochs: {epochs}\n")
        
        best_val_acc = 0.0
        
        for epoch in range(start_epoch, epochs):
            epoch_start = time.time()
            
            print(f"\n{'='*50}")
            print(f"Epoch {epoch + 1}/{epochs}")
            print(f"{'='*50}")
            
            # Train
            train_metrics = self.train_epoch(train_loader, epoch + 1)
            
            # Validate
            val_metrics, confusion_mat = self.validate(val_loader, epoch + 1)
            
            # Get current learning rate
            current_lr = self.optimizer.param_groups[0]["lr"]
            
            # Print epoch summary
            epoch_time = time.time() - epoch_start
            print(f"\nEpoch {epoch + 1} Summary ({epoch_time:.1f}s):")
            print(f"  Train Loss: {train_metrics['loss']:.4f} | Train Acc: {train_metrics['accuracy']:.4f}")
            print(f"  Val Loss: {val_metrics['loss']:.4f} | Val Acc: {val_metrics['accuracy']:.4f}")
            print(f"  Val F1: {val_metrics['f1_score']:.4f} | LR: {current_lr:.6f}")
            
            # Update history
            self.history["train_loss"].append(train_metrics["loss"])
            self.history["train_accuracy"].append(train_metrics["accuracy"])
            self.history["val_loss"].append(val_metrics["loss"])
            self.history["val_accuracy"].append(val_metrics["accuracy"])
            self.history["learning_rate"].append(current_lr)
            
            # Tensorboard logging
            if self.writer:
                self._log_to_tensorboard(epoch, train_metrics, val_metrics, confusion_mat)
            
            # Learning rate scheduler step
            if self.lr_callback:
                self.lr_callback.step(epoch, self.optimizer, val_metrics.get("loss"))
            
            # Save checkpoint
            self.checkpoint(
                model=self.model,
                optimizer=self.optimizer,
                epoch=epoch,
                metrics={"val_accuracy": val_metrics["accuracy"], "val_loss": val_metrics["loss"]},
                scheduler=self.scheduler
            )
            
            # Track best
            if val_metrics["accuracy"] > best_val_acc:
                best_val_acc = val_metrics["accuracy"]
            
            # Early stopping
            if self.early_stopping(val_metrics["accuracy"], epoch):
                print(f"\nEarly stopping at epoch {epoch + 1}")
                break
        
        print(f"\nTraining complete! Best validation accuracy: {best_val_acc:.4f}")
        
        if self.writer:
            self.writer.close()
        
        return self.history
    
    def _log_to_tensorboard(
        self,
        epoch: int,
        train_metrics: Dict[str, float],
        val_metrics: Dict[str, float],
        confusion_mat: np.ndarray
    ):
        """Log metrics to TensorBoard."""
        # Loss and accuracy
        self.writer.add_scalars("Loss", {
            "train": train_metrics["loss"],
            "val": val_metrics["loss"]
        }, epoch)
        
        self.writer.add_scalars("Accuracy", {
            "train": train_metrics["accuracy"],
            "val": val_metrics["accuracy"]
        }, epoch)
        
        # Per-class metrics
        for cls in self.class_names:
            if f"f1_{cls}" in val_metrics:
                self.writer.add_scalar(f"F1/{cls}", val_metrics[f"f1_{cls}"], epoch)
        
        # Learning rate
        self.writer.add_scalar(
            "Learning Rate",
            self.optimizer.param_groups[0]["lr"],
            epoch
        )
    
    def evaluate(
        self,
        test_loader: DataLoader
    ) -> Tuple[Dict[str, float], np.ndarray]:
        """
        Evaluate model on test set.
        
        Returns:
            Tuple of (metrics dictionary, confusion matrix)
        """
        return self.validate(test_loader, epoch=0)
    
    def predict(
        self,
        data_loader: DataLoader
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate predictions for a dataset.
        
        Returns:
            Tuple of (predictions, probabilities)
        """
        self.model.eval()
        
        all_preds = []
        all_probs = []
        
        with torch.no_grad():
            for images, _ in tqdm(data_loader, desc="Predicting"):
                images = images.to(self.device)
                
                if self.use_amp:
                    with autocast():
                        outputs = self.model(images)
                else:
                    outputs = self.model(images)
                
                probs = torch.softmax(outputs, dim=1)
                _, preds = torch.max(outputs, 1)
                
                all_preds.append(preds.cpu())
                all_probs.append(probs.cpu())
        
        return torch.cat(all_preds).numpy(), torch.cat(all_probs).numpy()
    
    def resume(self, checkpoint_path: str):
        """Resume training from a checkpoint."""
        checkpoint_info = ModelCheckpoint.load_checkpoint(
            checkpoint_path,
            self.model,
            self.optimizer,
            self.scheduler,
            device=str(self.device)
        )
        
        print(f"Resumed from epoch {checkpoint_info['epoch']}")
        print(f"Best score: {checkpoint_info['best_score']}")
        
        return checkpoint_info["epoch"] + 1


def create_trainer(
    model: nn.Module,
    config: Dict,
    device: torch.device,
    class_weights: Optional[torch.Tensor] = None
) -> Trainer:
    """
    Factory function to create a trainer with configuration.
    
    Args:
        model: Model to train
        config: Training configuration
        device: Device to train on
        class_weights: Optional class weights for loss function
        
    Returns:
        Configured Trainer instance
    """
    # Optimizer
    optimizer_name = config.get("optimizer", {}).get("name", "adamw")
    lr = config.get("optimizer", {}).get("lr", 1e-4)
    weight_decay = config.get("optimizer", {}).get("weight_decay", 0.01)
    
    if optimizer_name.lower() == "adamw":
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=lr,
            weight_decay=weight_decay
        )
    elif optimizer_name.lower() == "adam":
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=lr,
            weight_decay=weight_decay
        )
    else:
        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=lr,
            momentum=0.9,
            weight_decay=weight_decay
        )
    
    # Loss function
    loss_config = config.get("loss", {})
    loss_name = loss_config.get("name", "focal")
    
    if class_weights is not None:
        class_weights = class_weights.to(device)
    
    criterion = get_loss_function(
        loss_name,
        class_weights=class_weights,
        gamma=loss_config.get("gamma", 2.0),
        label_smoothing=loss_config.get("label_smoothing", 0.0)
    )
    
    # Learning rate scheduler
    scheduler_config = config.get("scheduler", {})
    scheduler_name = scheduler_config.get("name", "cosine_annealing_warm_restarts")
    
    if scheduler_name == "cosine_annealing_warm_restarts":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer,
            T_0=scheduler_config.get("T_0", 10),
            T_mult=scheduler_config.get("T_mult", 2),
            eta_min=scheduler_config.get("eta_min", 1e-6)
        )
    elif scheduler_name == "step":
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=scheduler_config.get("step_size", 10),
            gamma=scheduler_config.get("gamma", 0.1)
        )
    else:
        scheduler = None
    
    # Create trainer
    trainer_config = {
        "mixed_precision": config.get("mixed_precision", True),
        "gradient_accumulation_steps": config.get("training", {}).get("gradient_accumulation_steps", 1),
        "early_stopping_patience": config.get("training", {}).get("early_stopping_patience", 10),
        "checkpoint_dir": config.get("checkpoint", {}).get("save_dir", "checkpoints"),
        "log_dir": config.get("logging", {}).get("log_dir", "outputs/logs"),
        "tensorboard": config.get("logging", {}).get("tensorboard", True),
        "class_names": config.get("classes", ["A+", "A-", "B+", "B-", "AB+", "AB-", "O+", "O-"]),
    }
    
    return Trainer(
        model=model,
        optimizer=optimizer,
        criterion=criterion,
        device=device,
        scheduler=scheduler,
        config=trainer_config
    )
