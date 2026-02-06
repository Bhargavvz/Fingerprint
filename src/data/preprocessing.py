"""
Preprocessing utilities for fingerprint images.
Includes image enhancement and preparation for model input.
"""

from typing import Optional, Tuple, Union

import cv2
import numpy as np
from PIL import Image
import torch
from torchvision import transforms


def enhance_fingerprint(
    image: Union[np.ndarray, Image.Image],
    apply_clahe: bool = True,
    apply_gabor: bool = False,
    denoise: bool = True
) -> np.ndarray:
    """
    Enhance fingerprint image quality for better feature extraction.
    
    Args:
        image: Input fingerprint image
        apply_clahe: Apply Contrast Limited Adaptive Histogram Equalization
        apply_gabor: Apply Gabor filter for ridge enhancement
        denoise: Apply denoising
        
    Returns:
        Enhanced image as numpy array
    """
    # Convert to numpy if PIL Image
    if isinstance(image, Image.Image):
        image = np.array(image)
    
    # Convert to grayscale if needed
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    else:
        gray = image.copy()
    
    # Denoising
    if denoise:
        gray = cv2.fastNlMeansDenoising(gray, h=10)
    
    # CLAHE for contrast enhancement
    if apply_clahe:
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        gray = clahe.apply(gray)
    
    # Gabor filter for ridge enhancement (optional, computationally expensive)
    if apply_gabor:
        gray = apply_gabor_filter(gray)
    
    # Convert back to RGB
    enhanced = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
    
    return enhanced


def apply_gabor_filter(
    image: np.ndarray,
    ksize: int = 31,
    sigma: float = 4.0,
    lambd: float = 10.0,
    gamma: float = 0.5,
    psi: float = 0
) -> np.ndarray:
    """
    Apply Gabor filters at multiple orientations for ridge enhancement.
    """
    orientations = np.arange(0, np.pi, np.pi / 8)
    
    filtered_images = []
    for theta in orientations:
        kernel = cv2.getGaborKernel(
            (ksize, ksize),
            sigma=sigma,
            theta=theta,
            lambd=lambd,
            gamma=gamma,
            psi=psi
        )
        filtered = cv2.filter2D(image, cv2.CV_8UC1, kernel)
        filtered_images.append(filtered)
    
    # Take maximum response across orientations
    enhanced = np.max(np.stack(filtered_images), axis=0)
    
    return enhanced.astype(np.uint8)


def normalize_image(
    image: np.ndarray,
    target_mean: float = 128.0,
    target_var: float = 64.0
) -> np.ndarray:
    """
    Normalize image to have target mean and variance.
    """
    image = image.astype(np.float32)
    
    mean = np.mean(image)
    var = np.var(image)
    
    if var > 0:
        normalized = (image - mean) / np.sqrt(var) * np.sqrt(target_var) + target_mean
        normalized = np.clip(normalized, 0, 255).astype(np.uint8)
    else:
        normalized = image.astype(np.uint8)
    
    return normalized


def segment_fingerprint(
    image: np.ndarray,
    threshold_block_size: int = 35
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Segment fingerprint region from background.
    
    Returns:
        Tuple of (segmented_image, mask)
    """
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    else:
        gray = image.copy()
    
    # Compute local variance for segmentation
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Adaptive thresholding
    mask = cv2.adaptiveThreshold(
        blurred,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        threshold_block_size,
        2
    )
    
    # Morphological operations to clean up mask
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    
    # Apply mask to image
    if len(image.shape) == 3:
        segmented = cv2.bitwise_and(image, image, mask=mask)
    else:
        segmented = cv2.bitwise_and(gray, gray, mask=mask)
    
    return segmented, mask


def resize_with_padding(
    image: Union[np.ndarray, Image.Image],
    target_size: Tuple[int, int] = (224, 224),
    padding_color: Tuple[int, int, int] = (128, 128, 128)
) -> np.ndarray:
    """
    Resize image while maintaining aspect ratio, with padding.
    """
    if isinstance(image, Image.Image):
        image = np.array(image)
    
    h, w = image.shape[:2]
    target_h, target_w = target_size
    
    # Calculate scale factor
    scale = min(target_w / w, target_h / h)
    new_w, new_h = int(w * scale), int(h * scale)
    
    # Resize
    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
    
    # Create padded image
    if len(image.shape) == 3:
        padded = np.full((target_h, target_w, 3), padding_color, dtype=np.uint8)
    else:
        padded = np.full((target_h, target_w), padding_color[0], dtype=np.uint8)
    
    # Center the resized image
    y_offset = (target_h - new_h) // 2
    x_offset = (target_w - new_w) // 2
    
    padded[y_offset:y_offset + new_h, x_offset:x_offset + new_w] = resized
    
    return padded


def preprocess_image(
    image: Union[str, np.ndarray, Image.Image],
    target_size: Tuple[int, int] = (224, 224),
    enhance: bool = True,
    normalize: bool = True,
    to_tensor: bool = True
) -> Union[np.ndarray, torch.Tensor]:
    """
    Complete preprocessing pipeline for a single image.
    
    Args:
        image: Path to image, numpy array, or PIL Image
        target_size: Target size (height, width)
        enhance: Apply fingerprint enhancement
        normalize: Apply ImageNet normalization  
        to_tensor: Convert to PyTorch tensor
        
    Returns:
        Preprocessed image as numpy array or tensor
    """
    # Load image if path is provided
    if isinstance(image, str):
        image = Image.open(image).convert("RGB")
    elif isinstance(image, np.ndarray):
        if len(image.shape) == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        elif image.shape[2] == 4:
            image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
    
    # Convert to numpy
    if isinstance(image, Image.Image):
        image = np.array(image)
    
    # Enhance fingerprint
    if enhance:
        image = enhance_fingerprint(image)
    
    # Resize with padding
    image = resize_with_padding(image, target_size)
    
    # Convert to tensor if requested
    if to_tensor:
        image = Image.fromarray(image)
        transform_list = [transforms.ToTensor()]
        
        if normalize:
            transform_list.append(
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                )
            )
        
        transform = transforms.Compose(transform_list)
        image = transform(image)
    
    return image


def batch_preprocess(
    images: list,
    target_size: Tuple[int, int] = (224, 224),
    enhance: bool = True,
    device: str = "cpu"
) -> torch.Tensor:
    """
    Preprocess a batch of images.
    
    Args:
        images: List of image paths, arrays, or PIL Images
        target_size: Target size
        enhance: Apply enhancement
        device: Target device
        
    Returns:
        Batch tensor of shape (N, C, H, W)
    """
    processed = []
    
    for img in images:
        tensor = preprocess_image(
            img,
            target_size=target_size,
            enhance=enhance,
            normalize=True,
            to_tensor=True
        )
        processed.append(tensor)
    
    batch = torch.stack(processed)
    
    return batch.to(device)


if __name__ == "__main__":
    # Test preprocessing
    import matplotlib.pyplot as plt
    
    # Load sample image
    sample_path = "Dataset/A+/cluster_0_1001.BMP"
    original = Image.open(sample_path).convert("RGB")
    original_np = np.array(original)
    
    # Test enhancement
    enhanced = enhance_fingerprint(original_np)
    
    # Test full preprocessing
    preprocessed = preprocess_image(sample_path, enhance=True, to_tensor=False)
    
    # Visualize
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    
    axes[0].imshow(original_np)
    axes[0].set_title("Original")
    axes[0].axis("off")
    
    axes[1].imshow(enhanced)
    axes[1].set_title("Enhanced")
    axes[1].axis("off")
    
    axes[2].imshow(preprocessed)
    axes[2].set_title("Preprocessed (224x224)")
    axes[2].axis("off")
    
    plt.tight_layout()
    plt.savefig("outputs/preprocessing_test.png")
    plt.show()
    
    # Test tensor output
    tensor = preprocess_image(sample_path, to_tensor=True)
    print(f"Tensor shape: {tensor.shape}")
    print(f"Tensor range: [{tensor.min():.2f}, {tensor.max():.2f}]")
