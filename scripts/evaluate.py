#!/usr/bin/env python
"""
Evaluation script for trained models.

Usage:
    python scripts/evaluate.py --checkpoint checkpoints/best.pt
    python scripts/evaluate.py --checkpoint checkpoints/best.pt --data-dir Dataset
"""

import argparse
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import torch
import numpy as np

from src.data import create_data_loaders, get_val_transforms
from src.models import create_model
from src.training.callbacks import ModelCheckpoint
from src.training.metrics import compute_metrics, get_classification_report, compute_confusion_matrix
from src.utils import load_config, get_device
from src.utils.visualization import plot_confusion_matrix, save_figure


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate trained model")
    
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to model checkpoint"
    )
    
    parser.add_argument(
        "--data-dir",
        type=str,
        default="Dataset",
        help="Path to dataset directory"
    )
    
    parser.add_argument(
        "--config",
        type=str,
        default="configs/training_config.yaml",
        help="Path to configuration file"
    )
    
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs/evaluation",
        help="Output directory for results"
    )
    
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Device to use"
    )
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    print(f"\n{'='*60}")
    print("MODEL EVALUATION")
    print(f"{'='*60}")
    
    # Load config
    config = load_config(args.config)
    
    # Get device
    device = get_device(args.device)
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Get transforms
    image_size = config.get("data", {}).get("image_size", 224)
    val_transform = get_val_transforms(image_size=image_size)
    
    # Create data loaders
    print(f"\nLoading dataset from: {args.data_dir}")
    
    _, _, test_loader, full_dataset = create_data_loaders(
        root_dir=args.data_dir,
        train_transform=val_transform,
        val_transform=val_transform,
        test_transform=val_transform,
        batch_size=config["training"]["batch_size"],
        num_workers=config["data"].get("num_workers", 4),
    )
    
    # Create model
    print(f"\nCreating model: {config['model']['name']}")
    
    model = create_model(
        model_name=config["model"]["name"],
        num_classes=config["model"]["num_classes"],
        pretrained=False
    )
    
    # Load checkpoint
    print(f"Loading checkpoint: {args.checkpoint}")
    checkpoint_info = ModelCheckpoint.load_checkpoint(
        args.checkpoint,
        model,
        device=str(device)
    )
    print(f"  Loaded from epoch {checkpoint_info['epoch']}")
    
    model.to(device)
    model.eval()
    
    # Evaluate
    print("\nEvaluating on test set...")
    
    all_preds = []
    all_targets = []
    all_probs = []
    
    with torch.no_grad():
        for images, targets in test_loader:
            images = images.to(device)
            outputs = model(images)
            probs = torch.softmax(outputs, dim=1)
            _, preds = torch.max(outputs, 1)
            
            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(targets.numpy())
            all_probs.extend(probs.cpu().numpy())
    
    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)
    all_probs = np.array(all_probs)
    
    # Compute metrics
    class_names = config.get("classes", full_dataset.CLASSES)
    metrics = compute_metrics(all_preds, all_targets, class_names)
    
    print("\n" + "="*50)
    print("EVALUATION RESULTS")
    print("="*50)
    print(f"\nOverall Metrics:")
    print(f"  Accuracy:  {metrics['accuracy']:.4f}")
    print(f"  Precision: {metrics['precision']:.4f}")
    print(f"  Recall:    {metrics['recall']:.4f}")
    print(f"  F1-Score:  {metrics['f1_score']:.4f}")
    
    print("\nClassification Report:")
    print(get_classification_report(all_preds, all_targets, class_names))
    
    # Confusion matrix
    cm = compute_confusion_matrix(all_preds, all_targets)
    plot_confusion_matrix(
        cm,
        class_names,
        normalize=True,
        save_path=str(output_dir / "confusion_matrix.png")
    )
    
    # Save results
    results_path = output_dir / "evaluation_results.txt"
    with open(results_path, "w") as f:
        f.write("Model Evaluation Results\n")
        f.write("="*50 + "\n\n")
        f.write(f"Checkpoint: {args.checkpoint}\n")
        f.write(f"Dataset: {args.data_dir}\n\n")
        f.write("Metrics:\n")
        for key, value in metrics.items():
            if not key.startswith(("precision_", "recall_", "f1_")):
                f.write(f"  {key}: {value:.4f}\n")
        f.write("\nClassification Report:\n")
        f.write(get_classification_report(all_preds, all_targets, class_names))
    
    print(f"\nResults saved to {output_dir}")


if __name__ == "__main__":
    main()
