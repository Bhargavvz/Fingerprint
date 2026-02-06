"""
PyTorch Dataset class for Fingerprint Blood Group Detection.
Supports stratified train/val/test splits and K-Fold cross-validation.
"""

import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from PIL import Image
from sklearn.model_selection import StratifiedKFold, train_test_split
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler


class FingerprintDataset(Dataset):
    """
    PyTorch Dataset for fingerprint images with blood group labels.
    
    Args:
        image_paths: List of paths to fingerprint images
        labels: List of corresponding blood group labels (indices)
        transform: Optional transform to apply to images
        class_names: List of class names for reference
    """
    
    # Blood group class mapping
    CLASSES = ["A+", "A-", "B+", "B-", "AB+", "AB-", "O+", "O-"]
    CLASS_TO_IDX = {cls: idx for idx, cls in enumerate(CLASSES)}
    IDX_TO_CLASS = {idx: cls for idx, cls in enumerate(CLASSES)}
    
    def __init__(
        self,
        image_paths: List[str],
        labels: List[int],
        transform=None,
        class_names: Optional[List[str]] = None
    ):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform
        self.class_names = class_names or self.CLASSES
        
    def __len__(self) -> int:
        return len(self.image_paths)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        """Load and transform an image with its label."""
        image_path = self.image_paths[idx]
        label = self.labels[idx]
        
        # Load image
        image = Image.open(image_path).convert("RGB")
        
        # Apply transforms
        if self.transform:
            image = self.transform(image)
        
        return image, label
    
    def get_class_distribution(self) -> Dict[str, int]:
        """Get the distribution of classes in the dataset."""
        distribution = {}
        for label in self.labels:
            class_name = self.IDX_TO_CLASS[label]
            distribution[class_name] = distribution.get(class_name, 0) + 1
        return distribution
    
    @classmethod
    def from_directory(
        cls,
        root_dir: str,
        transform=None,
        extensions: Tuple[str, ...] = (".bmp", ".png", ".jpg", ".jpeg")
    ) -> "FingerprintDataset":
        """
        Create dataset from a directory structure.
        
        Expected structure:
            root_dir/
                A+/
                    image1.bmp
                    image2.bmp
                A-/
                    ...
        """
        image_paths = []
        labels = []
        
        root_path = Path(root_dir)
        
        for class_name in cls.CLASSES:
            class_dir = root_path / class_name
            if not class_dir.exists():
                print(f"Warning: Directory {class_dir} not found")
                continue
                
            for img_file in class_dir.iterdir():
                if img_file.suffix.lower() in extensions:
                    image_paths.append(str(img_file))
                    labels.append(cls.CLASS_TO_IDX[class_name])
        
        return cls(image_paths, labels, transform)


def create_stratified_split(
    dataset: FingerprintDataset,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    random_state: int = 42
) -> Tuple[List[int], List[int], List[int]]:
    """
    Create stratified train/val/test splits maintaining class distribution.
    
    Returns:
        Tuple of (train_indices, val_indices, test_indices)
    """
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, \
        "Ratios must sum to 1.0"
    
    indices = np.arange(len(dataset))
    labels = np.array(dataset.labels)
    
    # First split: train+val vs test
    train_val_idx, test_idx = train_test_split(
        indices,
        test_size=test_ratio,
        stratify=labels,
        random_state=random_state
    )
    
    # Second split: train vs val
    val_ratio_adjusted = val_ratio / (train_ratio + val_ratio)
    train_idx, val_idx = train_test_split(
        train_val_idx,
        test_size=val_ratio_adjusted,
        stratify=labels[train_val_idx],
        random_state=random_state
    )
    
    return train_idx.tolist(), val_idx.tolist(), test_idx.tolist()


def create_kfold_splits(
    dataset: FingerprintDataset,
    n_folds: int = 5,
    random_state: int = 42
) -> List[Tuple[List[int], List[int]]]:
    """
    Create K-Fold cross-validation splits.
    
    Returns:
        List of (train_indices, val_indices) tuples for each fold
    """
    kfold = StratifiedKFold(
        n_splits=n_folds,
        shuffle=True,
        random_state=random_state
    )
    
    indices = np.arange(len(dataset))
    labels = np.array(dataset.labels)
    
    splits = []
    for train_idx, val_idx in kfold.split(indices, labels):
        splits.append((train_idx.tolist(), val_idx.tolist()))
    
    return splits


def get_class_weights(dataset: FingerprintDataset) -> torch.Tensor:
    """
    Compute class weights inversely proportional to class frequencies.
    Used for handling class imbalance.
    """
    labels = np.array(dataset.labels)
    class_counts = np.bincount(labels, minlength=len(dataset.CLASSES))
    
    # Inverse frequency weighting
    weights = 1.0 / (class_counts + 1e-6)
    weights = weights / weights.sum() * len(dataset.CLASSES)
    
    return torch.FloatTensor(weights)


def get_sample_weights(dataset: FingerprintDataset) -> List[float]:
    """
    Get per-sample weights for WeightedRandomSampler.
    """
    class_weights = get_class_weights(dataset)
    sample_weights = [class_weights[label].item() for label in dataset.labels]
    return sample_weights


def create_data_loaders(
    root_dir: str,
    train_transform=None,
    val_transform=None,
    test_transform=None,
    batch_size: int = 32,
    num_workers: int = 4,
    pin_memory: bool = True,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    use_weighted_sampler: bool = True,
    random_state: int = 42
) -> Tuple[DataLoader, DataLoader, DataLoader, FingerprintDataset]:
    """
    Create train, validation, and test data loaders.
    
    Args:
        root_dir: Path to dataset root directory
        train_transform: Transform for training data
        val_transform: Transform for validation data
        test_transform: Transform for test data
        batch_size: Batch size for data loaders
        num_workers: Number of worker processes
        pin_memory: Whether to pin memory for faster GPU transfer
        train_ratio: Proportion of data for training
        val_ratio: Proportion of data for validation
        test_ratio: Proportion of data for testing
        use_weighted_sampler: Whether to use weighted sampling for training
        random_state: Random seed for reproducibility
        
    Returns:
        Tuple of (train_loader, val_loader, test_loader, full_dataset)
    """
    # Load full dataset without transforms
    full_dataset = FingerprintDataset.from_directory(root_dir)
    
    # Create stratified splits
    train_idx, val_idx, test_idx = create_stratified_split(
        full_dataset,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
        random_state=random_state
    )
    
    # Create subset datasets with appropriate transforms
    train_paths = [full_dataset.image_paths[i] for i in train_idx]
    train_labels = [full_dataset.labels[i] for i in train_idx]
    train_dataset = FingerprintDataset(train_paths, train_labels, train_transform)
    
    val_paths = [full_dataset.image_paths[i] for i in val_idx]
    val_labels = [full_dataset.labels[i] for i in val_idx]
    val_dataset = FingerprintDataset(val_paths, val_labels, val_transform)
    
    test_paths = [full_dataset.image_paths[i] for i in test_idx]
    test_labels = [full_dataset.labels[i] for i in test_idx]
    test_dataset = FingerprintDataset(test_paths, test_labels, test_transform)
    
    # Create sampler for training (handle class imbalance)
    train_sampler = None
    shuffle = True
    if use_weighted_sampler:
        sample_weights = get_sample_weights(train_dataset)
        train_sampler = WeightedRandomSampler(
            weights=sample_weights,
            num_samples=len(train_dataset),
            replacement=True
        )
        shuffle = False  # Sampler handles this
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        sampler=train_sampler,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory
    )
    
    # Print dataset statistics
    print(f"\nDataset Statistics:")
    print(f"  Total samples: {len(full_dataset)}")
    print(f"  Train samples: {len(train_dataset)}")
    print(f"  Val samples: {len(val_dataset)}")
    print(f"  Test samples: {len(test_dataset)}")
    print(f"\nClass Distribution (Full Dataset):")
    for cls, count in full_dataset.get_class_distribution().items():
        print(f"  {cls}: {count}")
    
    return train_loader, val_loader, test_loader, full_dataset


if __name__ == "__main__":
    # Test dataset loading
    import sys
    sys.path.append(str(Path(__file__).parent.parent.parent))
    
    from src.data.augmentation import get_train_transforms, get_val_transforms
    
    root_dir = "Dataset"
    
    train_loader, val_loader, test_loader, dataset = create_data_loaders(
        root_dir=root_dir,
        train_transform=get_train_transforms(),
        val_transform=get_val_transforms(),
        test_transform=get_val_transforms(),
        batch_size=32
    )
    
    # Test loading a batch
    images, labels = next(iter(train_loader))
    print(f"\nBatch shape: {images.shape}")
    print(f"Labels: {labels}")
