#!/usr/bin/env python3
"""
Advanced Training Script for Fingerprint Blood Group Detection
Optimized for multi-GPU training (2x NVIDIA T4) targeting 90%+ accuracy

Features:
- Multi-GPU DistributedDataParallel
- MixUp and CutMix augmentation
- Progressive image resizing
- Test-time augmentation (TTA)
- Automatic graph generation
- Git LFS model pushing
"""

import os
import sys
import argparse
import random
import time
import json
from pathlib import Path
from datetime import datetime

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
from torch.cuda.amp import GradScaler, autocast
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts, LinearLR, SequentialLR
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc
from sklearn.preprocessing import label_binarize
from tqdm import tqdm

# Project imports
from src.data import FingerprintDataset, get_train_transforms, get_val_transforms
from src.models import create_model
from src.training import FocalLoss, LabelSmoothingCrossEntropy
from src.utils import load_config, get_device, set_seed


def setup_distributed(rank, world_size):
    """Initialize distributed training."""
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)


def cleanup_distributed():
    """Clean up distributed training."""
    dist.destroy_process_group()


def mixup_data(x, y, alpha=0.4):
    """Apply MixUp augmentation."""
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size(0)
    index = torch.randperm(batch_size).to(x.device)
    
    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam


def cutmix_data(x, y, alpha=1.0):
    """Apply CutMix augmentation."""
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size(0)
    index = torch.randperm(batch_size).to(x.device)

    # Get bounding box
    W, H = x.size(2), x.size(3)
    cut_rat = np.sqrt(1. - lam)
    cut_w = int(W * cut_rat)
    cut_h = int(H * cut_rat)

    cx = np.random.randint(W)
    cy = np.random.randint(H)

    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)

    x[:, :, bbx1:bbx2, bby1:bby2] = x[index, :, bbx1:bbx2, bby1:bby2]
    
    # Adjust lambda based on actual box area
    lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (W * H))
    y_a, y_b = y, y[index]
    return x, y_a, y_b, lam


def mixup_criterion(criterion, pred, y_a, y_b, lam):
    """Compute MixUp/CutMix loss."""
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


class WarmupScheduler:
    """Warmup learning rate scheduler."""
    def __init__(self, optimizer, warmup_epochs, warmup_lr, base_lr, total_steps_per_epoch):
        self.optimizer = optimizer
        self.warmup_steps = warmup_epochs * total_steps_per_epoch
        self.warmup_lr = warmup_lr
        self.base_lr = base_lr
        self.step_count = 0
        
    def step(self):
        self.step_count += 1
        if self.step_count <= self.warmup_steps:
            lr = self.warmup_lr + (self.base_lr - self.warmup_lr) * (self.step_count / self.warmup_steps)
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = lr
            return True
        return False
    
    def get_lr(self):
        return self.optimizer.param_groups[0]['lr']


def save_training_curves(history, output_dir):
    """Save training curves as high-quality images."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Set style
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    epochs = range(1, len(history['train_loss']) + 1)
    
    # Loss curves
    ax = axes[0, 0]
    ax.plot(epochs, history['train_loss'], 'b-', linewidth=2, label='Training Loss', marker='o', markersize=4)
    ax.plot(epochs, history['val_loss'], 'r-', linewidth=2, label='Validation Loss', marker='s', markersize=4)
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Loss', fontsize=12)
    ax.set_title('Training and Validation Loss', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    
    # Accuracy curves
    ax = axes[0, 1]
    ax.plot(epochs, history['train_acc'], 'b-', linewidth=2, label='Training Accuracy', marker='o', markersize=4)
    ax.plot(epochs, history['val_acc'], 'r-', linewidth=2, label='Validation Accuracy', marker='s', markersize=4)
    ax.axhline(y=0.9, color='g', linestyle='--', linewidth=1.5, label='90% Target')
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Accuracy', fontsize=12)
    ax.set_title('Training and Validation Accuracy', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_ylim([0, 1.05])
    
    # Learning rate
    ax = axes[1, 0]
    ax.plot(epochs, history['lr'], 'g-', linewidth=2, marker='d', markersize=4)
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Learning Rate', fontsize=12)
    ax.set_title('Learning Rate Schedule', fontsize=14, fontweight='bold')
    ax.set_yscale('log')
    ax.grid(True, alpha=0.3)
    
    # F1 Score
    ax = axes[1, 1]
    if 'val_f1' in history:
        ax.plot(epochs, history['val_f1'], 'purple', linewidth=2, label='Validation F1', marker='^', markersize=4)
        ax.axhline(y=0.9, color='g', linestyle='--', linewidth=1.5, label='90% Target')
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('F1 Score', fontsize=12)
    ax.set_title('Validation F1 Score', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_ylim([0, 1.05])
    
    plt.tight_layout()
    plt.savefig(output_dir / 'training_curves.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ Saved training curves to {output_dir / 'training_curves.png'}")


def save_confusion_matrix(y_true, y_pred, class_names, output_dir):
    """Save confusion matrix as high-quality image."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    cm = confusion_matrix(y_true, y_pred)
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # Raw counts
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, 
                yticklabels=class_names, ax=axes[0], cbar_kws={'label': 'Count'})
    axes[0].set_xlabel('Predicted', fontsize=12)
    axes[0].set_ylabel('Actual', fontsize=12)
    axes[0].set_title('Confusion Matrix (Counts)', fontsize=14, fontweight='bold')
    
    # Normalized
    sns.heatmap(cm_normalized, annot=True, fmt='.2%', cmap='RdYlGn', xticklabels=class_names,
                yticklabels=class_names, ax=axes[1], vmin=0, vmax=1, cbar_kws={'label': 'Percentage'})
    axes[1].set_xlabel('Predicted', fontsize=12)
    axes[1].set_ylabel('Actual', fontsize=12)
    axes[1].set_title('Confusion Matrix (Normalized)', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'confusion_matrix.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ Saved confusion matrix to {output_dir / 'confusion_matrix.png'}")


def save_roc_curves(y_true, y_proba, class_names, output_dir):
    """Save ROC curves as high-quality image."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    n_classes = len(class_names)
    y_true_bin = label_binarize(y_true, classes=list(range(n_classes)))
    
    # Colors for blood groups
    colors = ['#ef4444', '#f97316', '#3b82f6', '#6366f1', '#8b5cf6', '#a855f7', '#22c55e', '#14b8a6']
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    for i, (class_name, color) in enumerate(zip(class_names, colors)):
        fpr, tpr, _ = roc_curve(y_true_bin[:, i], y_proba[:, i])
        roc_auc = auc(fpr, tpr)
        ax.plot(fpr, tpr, color=color, linewidth=2, label=f'{class_name} (AUC = {roc_auc:.3f})')
    
    ax.plot([0, 1], [0, 1], 'k--', linewidth=1.5, label='Random Classifier')
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate', fontsize=12)
    ax.set_ylabel('True Positive Rate', fontsize=12)
    ax.set_title('ROC Curves - Blood Group Classification', fontsize=14, fontweight='bold')
    ax.legend(loc='lower right', fontsize=9)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'roc_curves.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ Saved ROC curves to {output_dir / 'roc_curves.png'}")


def save_per_class_metrics(y_true, y_pred, class_names, output_dir):
    """Save per-class metrics as bar chart."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    report = classification_report(y_true, y_pred, target_names=class_names, output_dict=True)
    
    metrics = ['precision', 'recall', 'f1-score']
    x = np.arange(len(class_names))
    width = 0.25
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    colors = ['#3b82f6', '#22c55e', '#f59e0b']
    for i, metric in enumerate(metrics):
        values = [report[c][metric] for c in class_names]
        bars = ax.bar(x + i * width, values, width, label=metric.capitalize(), color=colors[i])
        
        # Add value labels
        for bar, val in zip(bars, values):
            ax.annotate(f'{val:.2f}', xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
                       xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=8)
    
    ax.axhline(y=0.9, color='red', linestyle='--', linewidth=1.5, label='90% Target')
    ax.set_xlabel('Blood Group', fontsize=12)
    ax.set_ylabel('Score', fontsize=12)
    ax.set_title('Per-Class Performance Metrics', fontsize=14, fontweight='bold')
    ax.set_xticks(x + width)
    ax.set_xticklabels(class_names, rotation=45, ha='right')
    ax.legend(fontsize=10)
    ax.set_ylim([0, 1.1])
    ax.grid(True, axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'per_class_metrics.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ Saved per-class metrics to {output_dir / 'per_class_metrics.png'}")
    
    # Also save as JSON
    with open(output_dir / 'classification_report.json', 'w') as f:
        json.dump(report, f, indent=2)
    print(f"✅ Saved classification report to {output_dir / 'classification_report.json'}")


def train_one_epoch(model, train_loader, criterion, optimizer, scaler, device, 
                    config, warmup_scheduler=None, epoch=0):
    """Train for one epoch with MixUp/CutMix."""
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    mixup_config = config.get('augmentation', {}).get('train', {}).get('mixup', {})
    cutmix_config = config.get('augmentation', {}).get('train', {}).get('cutmix', {})
    mix_prob = config.get('augmentation', {}).get('train', {}).get('mix_probability', 0.5)
    
    pbar = tqdm(train_loader, desc=f'Epoch {epoch}', leave=False)
    
    for batch_idx, (images, targets) in enumerate(pbar):
        images, targets = images.to(device), targets.to(device)
        
        # Apply MixUp or CutMix
        use_mix = np.random.random() < mix_prob
        if use_mix and (mixup_config.get('enabled') or cutmix_config.get('enabled')):
            if np.random.random() < 0.5 and mixup_config.get('enabled'):
                images, targets_a, targets_b, lam = mixup_data(images, targets, mixup_config.get('alpha', 0.4))
            elif cutmix_config.get('enabled'):
                images, targets_a, targets_b, lam = cutmix_data(images, targets, cutmix_config.get('alpha', 1.0))
            else:
                use_mix = False
        else:
            use_mix = False
        
        optimizer.zero_grad()
        
        with autocast():
            outputs = model(images)
            if use_mix:
                loss = mixup_criterion(criterion, outputs, targets_a, targets_b, lam)
            else:
                loss = criterion(outputs, targets)
        
        scaler.scale(loss).backward()
        
        # Gradient clipping
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), config['training'].get('gradient_clip_max_norm', 1.0))
        
        scaler.step(optimizer)
        scaler.update()
        
        # Warmup scheduler step
        if warmup_scheduler:
            warmup_scheduler.step()
        
        total_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        
        if use_mix:
            correct += (lam * predicted.eq(targets_a).sum().item() + 
                       (1 - lam) * predicted.eq(targets_b).sum().item())
        else:
            correct += predicted.eq(targets).sum().item()
        
        pbar.set_postfix({
            'loss': f'{total_loss / (batch_idx + 1):.4f}',
            'acc': f'{100. * correct / total:.2f}%'
        })
    
    return total_loss / len(train_loader), correct / total


def validate(model, val_loader, criterion, device, tta_enabled=False):
    """Validate model with optional TTA."""
    model.eval()
    total_loss = 0
    all_preds = []
    all_targets = []
    all_proba = []
    
    with torch.no_grad():
        for images, targets in tqdm(val_loader, desc='Validating', leave=False):
            images, targets = images.to(device), targets.to(device)
            
            if tta_enabled:
                # Test-time augmentation
                outputs_list = []
                
                # Original
                with autocast():
                    outputs_list.append(F.softmax(model(images), dim=1))
                
                # Horizontal flip
                with autocast():
                    outputs_list.append(F.softmax(model(torch.flip(images, dims=[3])), dim=1))
                
                # Average predictions
                outputs = torch.stack(outputs_list).mean(dim=0)
                loss = F.cross_entropy(outputs.log(), targets)
            else:
                with autocast():
                    outputs = model(images)
                loss = criterion(outputs, targets)
                outputs = F.softmax(outputs, dim=1)
            
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            
            all_preds.extend(predicted.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())
            all_proba.extend(outputs.cpu().numpy())
    
    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)
    all_proba = np.array(all_proba)
    
    accuracy = (all_preds == all_targets).mean()
    
    # Compute F1 score
    from sklearn.metrics import f1_score
    f1 = f1_score(all_targets, all_preds, average='macro')
    
    return total_loss / len(val_loader), accuracy, f1, all_preds, all_targets, all_proba


def main(rank=0, world_size=1, args=None):
    """Main training function."""
    # Load config
    config = load_config(args.config)
    
    # Setup distributed if enabled
    distributed = world_size > 1
    if distributed:
        setup_distributed(rank, world_size)
        device = torch.device(f'cuda:{rank}')
    else:
        device = get_device(config.get('device', 'auto'))
    
    is_main_process = rank == 0
    
    if is_main_process:
        print("=" * 60)
        print("🔬 ADVANCED FINGERPRINT BLOOD GROUP DETECTION TRAINING")
        print("=" * 60)
        print(f"📊 Device: {device}")
        print(f"🔢 World Size: {world_size}")
        print(f"📁 Config: {args.config}")
        print("=" * 60)
    
    # Set seed
    set_seed(config.get('seed', 42) + rank)
    
    # Create output directory
    output_dir = Path(config.get('outputs', {}).get('dir', 'outputs'))
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Get current image size for progressive resizing
    prog_config = config.get('augmentation', {}).get('progressive_resizing', {})
    if prog_config.get('enabled') and args.stage is not None:
        stages = prog_config.get('stages', [224])
        image_size = stages[min(args.stage, len(stages) - 1)]
    else:
        image_size = config['data'].get('image_size', 224)
    
    if is_main_process:
        print(f"📐 Image Size: {image_size}x{image_size}")
    
    # Create transforms
    train_transform = get_train_transforms(image_size=image_size)
    val_transform = get_val_transforms(image_size=image_size)
    
    # Create datasets
    from src.data import create_data_loaders
    
    # Get batch size (per GPU)
    batch_size = config['training']['batch_size']
    if distributed:
        batch_size = batch_size // world_size
    
    train_loader, val_loader, test_loader, dataset = create_data_loaders(
        root_dir=config['data']['root_dir'],
        train_transform=train_transform,
        val_transform=val_transform,
        test_transform=val_transform,
        batch_size=batch_size,
        num_workers=config['data'].get('num_workers', 4),
    )
    
    if distributed:
        train_sampler = DistributedSampler(train_loader.dataset, num_replicas=world_size, rank=rank)
        train_loader = DataLoader(
            train_loader.dataset,
            batch_size=batch_size,
            sampler=train_sampler,
            num_workers=config['data'].get('num_workers', 4),
            pin_memory=True,
        )
    
    class_names = config.get('classes', dataset.CLASSES)
    
    if is_main_process:
        print(f"📊 Training samples: {len(train_loader.dataset)}")
        print(f"📊 Validation samples: {len(val_loader.dataset)}")
        print(f"📊 Test samples: {len(test_loader.dataset)}")
        print(f"📊 Classes: {class_names}")
    
    # Create model
    model = create_model(
        model_name=config['model']['name'],
        num_classes=config['model']['num_classes'],
        pretrained=config['model']['pretrained'],
        dropout=config['model'].get('dropout', 0.3)
    )
    
    # Gradient checkpointing for memory efficiency
    if config['model'].get('use_gradient_checkpointing', False):
        if hasattr(model, 'set_grad_checkpointing'):
            model.set_grad_checkpointing(True)
    
    model = model.to(device)
    
    if distributed:
        model = DDP(model, device_ids=[rank])
    
    if is_main_process:
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"🧠 Model: {config['model']['name']}")
        print(f"🧠 Total Parameters: {total_params:,}")
        print(f"🧠 Trainable Parameters: {trainable_params:,}")
    
    # Create criterion
    loss_config = config.get('loss', {})
    if loss_config.get('name') == 'focal_loss':
        # Compute class weights
        class_counts = np.bincount([t for _, t in train_loader.dataset])
        class_weights = 1.0 / class_counts
        class_weights = class_weights / class_weights.sum() * len(class_weights)
        class_weights = torch.FloatTensor(class_weights).to(device)
        
        criterion = FocalLoss(
            gamma=loss_config.get('gamma', 2.0),
            alpha=class_weights,
            label_smoothing=loss_config.get('label_smoothing', 0.0)
        )
    else:
        criterion = LabelSmoothingCrossEntropy(
            smoothing=loss_config.get('label_smoothing', 0.1)
        )
    
    # Create optimizer
    opt_config = config['optimizer']
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=opt_config['lr'],
        weight_decay=opt_config.get('weight_decay', 0.01),
        betas=tuple(opt_config.get('betas', [0.9, 0.999]))
    )
    
    # Create scheduler
    sched_config = config['scheduler']
    scheduler = CosineAnnealingWarmRestarts(
        optimizer,
        T_0=sched_config.get('T_0', 10),
        T_mult=sched_config.get('T_mult', 2),
        eta_min=sched_config.get('eta_min', 1e-6)
    )
    
    # Warmup scheduler
    warmup_epochs = sched_config.get('warmup_epochs', 0)
    warmup_scheduler = None
    if warmup_epochs > 0:
        warmup_scheduler = WarmupScheduler(
            optimizer,
            warmup_epochs=warmup_epochs,
            warmup_lr=sched_config.get('warmup_lr', 1e-5),
            base_lr=opt_config['lr'],
            total_steps_per_epoch=len(train_loader)
        )
    
    # Mixed precision
    scaler = GradScaler() if config.get('mixed_precision', True) else None
    
    # Training history
    history = {
        'train_loss': [], 'train_acc': [],
        'val_loss': [], 'val_acc': [], 'val_f1': [],
        'lr': []
    }
    
    # Training loop
    best_val_f1 = 0
    patience_counter = 0
    epochs = config['training']['epochs']
    patience = config['training'].get('early_stopping_patience', 15)
    
    checkpoint_dir = Path(config['checkpoint']['save_dir'])
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    if is_main_process:
        print("\n🚀 Starting Training...")
        print("-" * 60)
    
    for epoch in range(1, epochs + 1):
        if distributed:
            train_sampler.set_epoch(epoch)
        
        # Train
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, scaler, device,
            config, warmup_scheduler if epoch <= warmup_epochs else None, epoch
        )
        
        # Validate
        tta_enabled = config.get('augmentation', {}).get('tta', {}).get('enabled', False)
        val_loss, val_acc, val_f1, val_preds, val_targets, val_proba = validate(
            model, val_loader, criterion, device, tta_enabled=tta_enabled
        )
        
        # Update scheduler (after warmup)
        if epoch > warmup_epochs:
            scheduler.step()
        
        current_lr = optimizer.param_groups[0]['lr']
        
        # Update history
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        history['val_f1'].append(val_f1)
        history['lr'].append(current_lr)
        
        if is_main_process:
            print(f"Epoch {epoch:3d}/{epochs} | "
                  f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | "
                  f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f} | Val F1: {val_f1:.4f} | "
                  f"LR: {current_lr:.6f}")
            
            # Save best model
            if val_f1 > best_val_f1:
                best_val_f1 = val_f1
                patience_counter = 0
                
                state = {
                    'epoch': epoch,
                    'model_state_dict': model.module.state_dict() if distributed else model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'val_f1': val_f1,
                    'val_acc': val_acc,
                    'config': config
                }
                torch.save(state, checkpoint_dir / 'best_model.pt')
                print(f"  ✅ New best model saved! F1: {val_f1:.4f}")
            else:
                patience_counter += 1
            
            # Save periodic checkpoint
            if epoch % config['checkpoint'].get('save_every_n_epochs', 10) == 0:
                state = {
                    'epoch': epoch,
                    'model_state_dict': model.module.state_dict() if distributed else model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'val_f1': val_f1,
                    'val_acc': val_acc,
                    'history': history
                }
                torch.save(state, checkpoint_dir / f'checkpoint_epoch_{epoch}.pt')
            
            # Check early stopping
            if patience_counter >= patience:
                print(f"\n⏹️  Early stopping triggered after {patience} epochs without improvement")
                break
    
    # Final evaluation on test set
    if is_main_process:
        print("\n" + "=" * 60)
        print("📊 FINAL EVALUATION ON TEST SET")
        print("=" * 60)
        
        # Load best model
        checkpoint = torch.load(checkpoint_dir / 'best_model.pt', weights_only=False)
        if distributed:
            model.module.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint['model_state_dict'])
        
        # Evaluate
        tta_enabled = config.get('augmentation', {}).get('tta', {}).get('enabled', False)
        test_loss, test_acc, test_f1, test_preds, test_targets, test_proba = validate(
            model, test_loader, criterion, device, tta_enabled=tta_enabled
        )
        
        print(f"\n🎯 Test Accuracy: {test_acc:.4f} ({test_acc*100:.2f}%)")
        print(f"🎯 Test F1 Score: {test_f1:.4f} ({test_f1*100:.2f}%)")
        
        # Check if we hit 90%+ target
        if test_acc >= 0.90:
            print("\n🎉 " + "=" * 50)
            print("🎉 CONGRATULATIONS! TARGET 90%+ ACCURACY ACHIEVED!")
            print("🎉 " + "=" * 50)
        
        # Save all outputs
        print("\n📁 Saving output files...")
        
        save_training_curves(history, output_dir)
        save_confusion_matrix(test_targets, test_preds, class_names, output_dir)
        save_roc_curves(test_targets, test_proba, class_names, output_dir)
        save_per_class_metrics(test_targets, test_preds, class_names, output_dir)
        
        # Save final metrics
        final_metrics = {
            'test_accuracy': float(test_acc),
            'test_f1_score': float(test_f1),
            'test_loss': float(test_loss),
            'best_val_f1': float(best_val_f1),
            'total_epochs': epoch,
            'config': args.config,
            'timestamp': datetime.now().isoformat()
        }
        
        with open(output_dir / 'final_metrics.json', 'w') as f:
            json.dump(final_metrics, f, indent=2)
        
        print(f"\n✅ All outputs saved to {output_dir}/")
        print("=" * 60)
    
    if distributed:
        cleanup_distributed()


def parse_args():
    parser = argparse.ArgumentParser(description='Advanced Training Script')
    parser.add_argument('--config', type=str, default='configs/training_config.yaml',
                       help='Path to config file')
    parser.add_argument('--stage', type=int, default=None,
                       help='Progressive resizing stage (0, 1, 2)')
    parser.add_argument('--resume', type=str, default=None,
                       help='Path to checkpoint to resume from')
    parser.add_argument('--gpus', type=int, default=1,
                       help='Number of GPUs to use')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    
    if args.gpus > 1:
        import torch.multiprocessing as mp
        mp.spawn(main, args=(args.gpus, args), nprocs=args.gpus, join=True)
    else:
        main(rank=0, world_size=1, args=args)
