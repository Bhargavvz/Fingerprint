"""
Data augmentation transforms for fingerprint images.
Uses torchvision transforms and albumentations for advanced augmentations.
"""

from typing import Tuple

import numpy as np
import torch
from PIL import Image
from torchvision import transforms
import albumentations as A
from albumentations.pytorch import ToTensorV2


# ImageNet normalization statistics
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


class AlbumentationsTransform:
    """Wrapper to use albumentations with PyTorch Dataset."""
    
    def __init__(self, transform: A.Compose):
        self.transform = transform
    
    def __call__(self, img: Image.Image) -> torch.Tensor:
        # Convert PIL Image to numpy array
        img_np = np.array(img)
        
        # Apply albumentations transform
        augmented = self.transform(image=img_np)
        
        return augmented["image"]


def get_train_transforms(
    image_size: int = 224,
    rotation_limit: int = 15,
    brightness_limit: float = 0.2,
    contrast_limit: float = 0.2
) -> AlbumentationsTransform:
    """
    Get training transforms with data augmentation.
    
    Includes:
    - Resize and padding
    - Random horizontal flip
    - Random rotation
    - Brightness/contrast adjustment
    - Gaussian blur
    - Elastic transformation (good for fingerprints)
    - Normalization
    """
    transform = A.Compose([
        # Resize with padding to maintain aspect ratio
        A.LongestMaxSize(max_size=image_size),
        A.PadIfNeeded(
            min_height=image_size,
            min_width=image_size,
            border_mode=0,
            fill=(128, 128, 128)
        ),
        
        # Geometric augmentations
        A.HorizontalFlip(p=0.5),
        A.Rotate(limit=rotation_limit, p=0.5, border_mode=0),
        A.Affine(
            translate_percent={"x": (-0.1, 0.1), "y": (-0.1, 0.1)},
            scale=(0.9, 1.1),
            rotate=0,
            p=0.3,
            mode=0
        ),
        
        # Elastic deformation - particularly useful for fingerprints
        A.ElasticTransform(
            alpha=50,
            sigma=10,
            p=0.3
        ),
        
        # Color augmentations
        A.OneOf([
            A.RandomBrightnessContrast(
                brightness_limit=brightness_limit,
                contrast_limit=contrast_limit,
                p=1.0
            ),
            A.RandomGamma(gamma_limit=(80, 120), p=1.0),
        ], p=0.5),
        
        # Blur augmentations
        A.OneOf([
            A.GaussianBlur(blur_limit=(3, 5), p=1.0),
            A.MotionBlur(blur_limit=3, p=1.0),
        ], p=0.2),
        
        # Noise - using std_range instead of deprecated var_limit
        A.GaussNoise(std_range=(0.02, 0.1), p=0.2),
        
        # Fingerprint-specific: enhance ridge patterns
        A.OneOf([
            A.CLAHE(clip_limit=2.0, tile_grid_size=(8, 8), p=1.0),
            A.Sharpen(alpha=(0.2, 0.5), lightness=(0.5, 1.0), p=1.0),
        ], p=0.3),
        
        # Normalize and convert to tensor
        A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ToTensorV2()
    ])
    
    return AlbumentationsTransform(transform)


def get_val_transforms(image_size: int = 224) -> AlbumentationsTransform:
    """
    Get validation/test transforms (no augmentation).
    
    Only includes:
    - Resize
    - Normalization
    """
    transform = A.Compose([
        # Resize with padding
        A.LongestMaxSize(max_size=image_size),
        A.PadIfNeeded(
            min_height=image_size,
            min_width=image_size,
            border_mode=0,
            fill=(128, 128, 128)
        ),
        
        # Normalize and convert to tensor
        A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ToTensorV2()
    ])
    
    return AlbumentationsTransform(transform)


def get_test_transforms(image_size: int = 224) -> AlbumentationsTransform:
    """
    Get test transforms (same as validation).
    """
    return get_val_transforms(image_size)


def get_tta_transforms(image_size: int = 224) -> list:
    """
    Get Test Time Augmentation (TTA) transforms.
    Returns a list of transforms to apply during inference.
    """
    base_transform = get_val_transforms(image_size)
    
    # TTA: original + horizontal flip
    tta_transforms = [
        base_transform,
        AlbumentationsTransform(A.Compose([
            A.LongestMaxSize(max_size=image_size),
            A.PadIfNeeded(
                min_height=image_size,
                min_width=image_size,
                border_mode=0,
                fill=(128, 128, 128)
            ),
            A.HorizontalFlip(p=1.0),
            A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ToTensorV2()
        ]))
    ]
    
    return tta_transforms


def denormalize(
    tensor: torch.Tensor,
    mean: Tuple[float, ...] = IMAGENET_MEAN,
    std: Tuple[float, ...] = IMAGENET_STD
) -> torch.Tensor:
    """
    Denormalize a tensor image for visualization.
    """
    mean = torch.tensor(mean).view(-1, 1, 1)
    std = torch.tensor(std).view(-1, 1, 1)
    
    if tensor.device.type == "cuda":
        mean = mean.cuda()
        std = std.cuda()
    
    return tensor * std + mean


if __name__ == "__main__":
    # Test augmentations
    from PIL import Image
    import matplotlib.pyplot as plt
    
    # Load a sample image
    sample_img = Image.open("Dataset/A+/cluster_0_1001.BMP").convert("RGB")
    
    # Get transforms
    train_transform = get_train_transforms()
    val_transform = get_val_transforms()
    
    # Apply transforms
    train_img = train_transform(sample_img)
    val_img = val_transform(sample_img)
    
    print(f"Train image shape: {train_img.shape}")
    print(f"Val image shape: {val_img.shape}")
    
    # Visualize
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    
    axes[0].imshow(sample_img)
    axes[0].set_title("Original")
    axes[0].axis("off")
    
    # Denormalize for visualization
    train_vis = denormalize(train_img).permute(1, 2, 0).numpy().clip(0, 1)
    axes[1].imshow(train_vis)
    axes[1].set_title("Train Augmented")
    axes[1].axis("off")
    
    val_vis = denormalize(val_img).permute(1, 2, 0).numpy().clip(0, 1)
    axes[2].imshow(val_vis)
    axes[2].set_title("Validation")
    axes[2].axis("off")
    
    plt.tight_layout()
    plt.savefig("outputs/augmentation_test.png")
    plt.show()
