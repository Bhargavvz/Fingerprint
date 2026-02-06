#!/usr/bin/env python
"""
Main training script for Fingerprint Blood Group Detection.

Usage:
    python scripts/train.py
    python scripts/train.py --config configs/training_config.yaml
    python scripts/train.py --epochs 100 --batch-size 64
    python scripts/train.py --resume checkpoints/last.pt
"""

import argparse
import os
import sys
from datetime import datetime
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import torch

from src.data import create_data_loaders, get_train_transforms, get_val_transforms, get_class_weights
from src.models import create_model
from src.training import create_trainer
from src.training.callbacks import ModelCheckpoint
from src.training.metrics import get_classification_report, compute_confusion_matrix
from src.utils import load_config, get_device, set_seed
from src.utils.visualization import (
    plot_training_history,
    plot_confusion_matrix,
    plot_class_distribution
)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Train Fingerprint Blood Group Detection Model"
    )
    
    parser.add_argument(
        "--config",
        type=str,
        default="configs/training_config.yaml",
        help="Path to configuration file"
    )
    
    parser.add_argument(
        "--data-dir",
        type=str,
        default="Dataset",
        help="Path to dataset directory"
    )
    
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Number of training epochs (overrides config)"
    )
    
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Batch size (overrides config)"
    )
    
    parser.add_argument(
        "--lr",
        type=float,
        default=None,
        help="Learning rate (overrides config)"
    )
    
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Path to checkpoint to resume from"
    )
    
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Device to use: auto, cuda, cpu"
    )
    
    parser.add_argument(
        "--test-run",
        action="store_true",
        help="Quick test run with minimal epochs"
    )
    
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs",
        help="Output directory for logs and figures"
    )
    
    return parser.parse_args()


def main():
    """Main training function."""
    args = parse_args()
    
    # Load configuration
    print(f"\n{'='*60}")
    print("FINGERPRINT BLOOD GROUP DETECTION - TRAINING")
    print(f"{'='*60}")
    print(f"Configuration: {args.config}")
    
    config = load_config(args.config)
    
    # Override config with command line arguments
    if args.epochs is not None:
        config["training"]["epochs"] = args.epochs
    if args.batch_size is not None:
        config["training"]["batch_size"] = args.batch_size
    if args.lr is not None:
        config["optimizer"]["lr"] = args.lr
    
    # Test run settings
    if args.test_run:
        config["training"]["epochs"] = 2
        config["training"]["batch_size"] = 4
        print("\n*** TEST RUN MODE - Minimal training ***")
    
    # Set random seed
    seed = config.get("seed", 42)
    set_seed(seed)
    
    # Get device
    device = get_device(args.device)
    
    # Create output directories
    output_dir = Path(args.output_dir)
    (output_dir / "logs").mkdir(parents=True, exist_ok=True)
    (output_dir / "figures").mkdir(parents=True, exist_ok=True)
    Path("checkpoints").mkdir(parents=True, exist_ok=True)
    
    # Get transforms
    image_size = config.get("data", {}).get("image_size", 224)
    train_transform = get_train_transforms(image_size=image_size)
    val_transform = get_val_transforms(image_size=image_size)
    
    # Create data loaders
    print(f"\nLoading dataset from: {args.data_dir}")
    
    train_loader, val_loader, test_loader, full_dataset = create_data_loaders(
        root_dir=args.data_dir,
        train_transform=train_transform,
        val_transform=val_transform,
        test_transform=val_transform,
        batch_size=config["training"]["batch_size"],
        num_workers=config["data"].get("num_workers", 4),
        pin_memory=config["data"].get("pin_memory", True),
        train_ratio=config["data"].get("train_split", 0.7),
        val_ratio=config["data"].get("val_split", 0.15),
        test_ratio=config["data"].get("test_split", 0.15),
        use_weighted_sampler=True,
        random_state=seed
    )
    
    # Plot class distribution
    class_dist = full_dataset.get_class_distribution()
    plot_class_distribution(
        class_dist,
        save_path=str(output_dir / "figures" / "class_distribution.png")
    )
    
    # Get class weights
    class_weights = get_class_weights(full_dataset)
    print(f"\nClass weights: {class_weights.tolist()}")
    
    # Create model
    print(f"\nCreating model: {config['model']['name']}")
    
    model = create_model(
        model_name=config["model"]["name"],
        num_classes=config["model"]["num_classes"],
        pretrained=config["model"]["pretrained"],
        dropout=config["model"].get("dropout", 0.3),
        attention_reduction=config["model"].get("attention_reduction", 16)
    )
    
    # Print model info
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    # Create trainer
    trainer = create_trainer(
        model=model,
        config=config,
        device=device,
        class_weights=class_weights
    )
    
    # Resume from checkpoint if specified
    start_epoch = 0
    if args.resume:
        print(f"\nResuming from checkpoint: {args.resume}")
        start_epoch = trainer.resume(args.resume)
    
    # Training
    print(f"\n{'='*60}")
    print("STARTING TRAINING")
    print(f"{'='*60}")
    print(f"Epochs: {config['training']['epochs']}")
    print(f"Batch size: {config['training']['batch_size']}")
    print(f"Learning rate: {config['optimizer']['lr']}")
    print(f"Device: {device}")
    
    history = trainer.fit(
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=config["training"]["epochs"],
        start_epoch=start_epoch
    )
    
    # Plot training history
    plot_training_history(
        history,
        save_path=str(output_dir / "figures" / "training_history.png")
    )
    
    # Final evaluation on test set
    print(f"\n{'='*60}")
    print("FINAL EVALUATION ON TEST SET")
    print(f"{'='*60}")
    
    # Load best model
    best_checkpoint = "checkpoints/best.pt"
    if os.path.exists(best_checkpoint):
        checkpoint_info = ModelCheckpoint.load_checkpoint(
            best_checkpoint,
            model,
            device=str(device)
        )
        print(f"Loaded best model from epoch {checkpoint_info['epoch']}")
    
    # Evaluate
    test_metrics, test_cm = trainer.evaluate(test_loader)
    
    print("\nTest Set Results:")
    print(f"  Accuracy: {test_metrics['accuracy']:.4f}")
    print(f"  Precision: {test_metrics['precision']:.4f}")
    print(f"  Recall: {test_metrics['recall']:.4f}")
    print(f"  F1-Score: {test_metrics['f1_score']:.4f}")
    
    # Plot confusion matrix
    class_names = config.get("classes", full_dataset.CLASSES)
    plot_confusion_matrix(
        test_cm,
        class_names,
        normalize=True,
        save_path=str(output_dir / "figures" / "confusion_matrix.png")
    )
    
    # Get predictions for classification report
    preds, probs = trainer.predict(test_loader)
    test_targets = []
    for _, targets in test_loader:
        test_targets.extend(targets.numpy())
    
    print("\nClassification Report:")
    print(get_classification_report(preds, test_targets, class_names))
    
    # Save final metrics
    metrics_path = output_dir / "final_metrics.txt"
    with open(metrics_path, "w") as f:
        f.write("Fingerprint Blood Group Detection - Final Results\n")
        f.write(f"{'='*50}\n\n")
        f.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Model: {config['model']['name']}\n\n")
        f.write("Test Set Metrics:\n")
        for key, value in test_metrics.items():
            if not key.startswith(("precision_", "recall_", "f1_")):
                f.write(f"  {key}: {value:.4f}\n")
        f.write(f"\nClassification Report:\n")
        f.write(get_classification_report(preds, test_targets, class_names))
    
    print(f"\nResults saved to {metrics_path}")
    print(f"Figures saved to {output_dir / 'figures'}")
    print(f"Checkpoints saved to checkpoints/")
    
    print(f"\n{'='*60}")
    print("TRAINING COMPLETE!")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
