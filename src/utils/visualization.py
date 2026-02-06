"""
Visualization utilities for training and evaluation.
"""

from pathlib import Path
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


# Set style
plt.style.use("seaborn-v0_8-whitegrid")
sns.set_palette("husl")


def plot_training_history(
    history: Dict[str, List[float]],
    save_path: Optional[str] = None,
    figsize: tuple = (14, 5)
):
    """
    Plot training history curves.
    
    Args:
        history: Dictionary with 'train_loss', 'val_loss', 'train_accuracy', 'val_accuracy'
        save_path: Optional path to save figure
        figsize: Figure size
    """
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    
    epochs = range(1, len(history["train_loss"]) + 1)
    
    # Loss plot
    axes[0].plot(epochs, history["train_loss"], "b-", label="Training Loss", linewidth=2)
    axes[0].plot(epochs, history["val_loss"], "r-", label="Validation Loss", linewidth=2)
    axes[0].set_xlabel("Epoch", fontsize=12)
    axes[0].set_ylabel("Loss", fontsize=12)
    axes[0].set_title("Training and Validation Loss", fontsize=14)
    axes[0].legend(fontsize=10)
    axes[0].grid(True, alpha=0.3)
    
    # Accuracy plot
    axes[1].plot(epochs, history["train_accuracy"], "b-", label="Training Accuracy", linewidth=2)
    axes[1].plot(epochs, history["val_accuracy"], "r-", label="Validation Accuracy", linewidth=2)
    axes[1].set_xlabel("Epoch", fontsize=12)
    axes[1].set_ylabel("Accuracy", fontsize=12)
    axes[1].set_title("Training and Validation Accuracy", fontsize=14)
    axes[1].legend(fontsize=10)
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        save_figure(fig, save_path)
    
    return fig


def plot_confusion_matrix(
    cm: np.ndarray,
    class_names: List[str],
    normalize: bool = True,
    save_path: Optional[str] = None,
    figsize: tuple = (10, 8),
    cmap: str = "Blues"
):
    """
    Plot confusion matrix.
    
    Args:
        cm: Confusion matrix array
        class_names: List of class names
        normalize: Whether to normalize the matrix
        save_path: Optional path to save figure
        figsize: Figure size
        cmap: Colormap name
    """
    if normalize:
        cm = cm.astype("float") / (cm.sum(axis=1, keepdims=True) + 1e-6)
    
    fig, ax = plt.subplots(figsize=figsize)
    
    sns.heatmap(
        cm,
        annot=True,
        fmt=".2f" if normalize else "d",
        cmap=cmap,
        xticklabels=class_names,
        yticklabels=class_names,
        ax=ax,
        square=True,
        linewidths=0.5,
        cbar_kws={"shrink": 0.8}
    )
    
    ax.set_xlabel("Predicted Label", fontsize=12)
    ax.set_ylabel("True Label", fontsize=12)
    ax.set_title("Confusion Matrix", fontsize=14)
    
    plt.tight_layout()
    
    if save_path:
        save_figure(fig, save_path)
    
    return fig


def plot_class_distribution(
    class_counts: Dict[str, int],
    save_path: Optional[str] = None,
    figsize: tuple = (10, 6)
):
    """
    Plot class distribution bar chart.
    
    Args:
        class_counts: Dictionary of class names to counts
        save_path: Optional path to save figure
        figsize: Figure size
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    classes = list(class_counts.keys())
    counts = list(class_counts.values())
    
    bars = ax.bar(classes, counts, color=sns.color_palette("husl", len(classes)))
    
    # Add value labels on bars
    for bar, count in zip(bars, counts):
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2.,
            height + max(counts) * 0.02,
            f"{count}",
            ha="center",
            va="bottom",
            fontsize=10
        )
    
    ax.set_xlabel("Blood Group", fontsize=12)
    ax.set_ylabel("Number of Samples", fontsize=12)
    ax.set_title("Class Distribution in Dataset", fontsize=14)
    
    # Add total count
    total = sum(counts)
    ax.text(
        0.98, 0.98,
        f"Total: {total}",
        transform=ax.transAxes,
        fontsize=11,
        verticalalignment="top",
        horizontalalignment="right",
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5)
    )
    
    plt.tight_layout()
    
    if save_path:
        save_figure(fig, save_path)
    
    return fig


def plot_predictions(
    images: np.ndarray,
    true_labels: List[str],
    pred_labels: List[str],
    confidences: List[float],
    save_path: Optional[str] = None,
    n_samples: int = 8,
    figsize: tuple = (16, 8)
):
    """
    Plot sample predictions with images.
    
    Args:
        images: Array of images (N, H, W, C)
        true_labels: List of true class names
        pred_labels: List of predicted class names
        confidences: List of prediction confidences
        save_path: Optional path to save figure
        n_samples: Number of samples to show
        figsize: Figure size
    """
    n_samples = min(n_samples, len(images))
    n_cols = 4
    n_rows = (n_samples + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    axes = axes.flatten() if n_samples > 1 else [axes]
    
    for i in range(n_samples):
        ax = axes[i]
        
        # Handle different image formats
        img = images[i]
        if img.max() <= 1.0:
            img = (img * 255).astype(np.uint8)
        
        ax.imshow(img)
        
        # Color based on correctness
        color = "green" if true_labels[i] == pred_labels[i] else "red"
        
        ax.set_title(
            f"True: {true_labels[i]}\nPred: {pred_labels[i]} ({confidences[i]:.1%})",
            color=color,
            fontsize=10
        )
        ax.axis("off")
    
    # Hide unused axes
    for i in range(n_samples, len(axes)):
        axes[i].axis("off")
    
    plt.tight_layout()
    
    if save_path:
        save_figure(fig, save_path)
    
    return fig


def plot_gradcam(
    original_image: np.ndarray,
    heatmap: np.ndarray,
    overlay: np.ndarray,
    class_name: str,
    confidence: float,
    save_path: Optional[str] = None,
    figsize: tuple = (15, 5)
):
    """
    Plot Grad-CAM visualization.
    
    Args:
        original_image: Original input image
        heatmap: Grad-CAM heatmap
        overlay: Overlay of heatmap on image
        class_name: Predicted class name
        confidence: Prediction confidence
        save_path: Optional path to save figure
        figsize: Figure size
    """
    fig, axes = plt.subplots(1, 3, figsize=figsize)
    
    # Original image
    axes[0].imshow(original_image)
    axes[0].set_title("Original Fingerprint", fontsize=12)
    axes[0].axis("off")
    
    # Heatmap
    axes[1].imshow(heatmap, cmap="jet")
    axes[1].set_title("Grad-CAM Heatmap", fontsize=12)
    axes[1].axis("off")
    
    # Overlay
    axes[2].imshow(overlay)
    axes[2].set_title(f"Prediction: {class_name} ({confidence:.1%})", fontsize=12)
    axes[2].axis("off")
    
    plt.suptitle("Explainable AI - Grad-CAM Visualization", fontsize=14)
    plt.tight_layout()
    
    if save_path:
        save_figure(fig, save_path)
    
    return fig


def plot_learning_rate(
    lr_history: List[float],
    save_path: Optional[str] = None,
    figsize: tuple = (10, 4)
):
    """
    Plot learning rate schedule.
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    epochs = range(1, len(lr_history) + 1)
    ax.plot(epochs, lr_history, "b-", linewidth=2)
    
    ax.set_xlabel("Epoch", fontsize=12)
    ax.set_ylabel("Learning Rate", fontsize=12)
    ax.set_title("Learning Rate Schedule", fontsize=14)
    ax.set_yscale("log")
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        save_figure(fig, save_path)
    
    return fig


def save_figure(fig, path: str, dpi: int = 150):
    """
    Save figure to file.
    
    Args:
        fig: Matplotlib figure
        path: Path to save figure
        dpi: Resolution
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    
    fig.savefig(path, dpi=dpi, bbox_inches="tight", facecolor="white")
    print(f"Saved figure to {path}")


if __name__ == "__main__":
    # Test visualizations
    import os
    
    # Create test data
    history = {
        "train_loss": np.random.rand(20).cumsum()[::-1] / 5,
        "val_loss": np.random.rand(20).cumsum()[::-1] / 4,
        "train_accuracy": np.linspace(0.5, 0.95, 20) + np.random.rand(20) * 0.05,
        "val_accuracy": np.linspace(0.45, 0.90, 20) + np.random.rand(20) * 0.05,
    }
    
    # Test plots
    os.makedirs("outputs/figures", exist_ok=True)
    
    fig = plot_training_history(history, "outputs/figures/training_history.png")
    plt.close(fig)
    
    cm = np.random.randint(0, 100, (8, 8))
    np.fill_diagonal(cm, np.random.randint(200, 300, 8))
    class_names = ["A+", "A-", "B+", "B-", "AB+", "AB-", "O+", "O-"]
    
    fig = plot_confusion_matrix(cm, class_names, save_path="outputs/figures/confusion_matrix.png")
    plt.close(fig)
    
    class_counts = {"A+": 565, "A-": 1009, "B+": 652, "B-": 741, "AB+": 708, "AB-": 761, "O+": 852, "O-": 712}
    fig = plot_class_distribution(class_counts, save_path="outputs/figures/class_distribution.png")
    plt.close(fig)
    
    print("Test visualizations saved to outputs/figures/")
