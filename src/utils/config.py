"""
Configuration management utilities.
"""

import os
from pathlib import Path
from typing import Any, Dict, Optional

import torch
import yaml


def load_config(config_path: str) -> Dict[str, Any]:
    """
    Load configuration from YAML file.
    
    Args:
        config_path: Path to configuration file
        
    Returns:
        Configuration dictionary
    """
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    
    return config


def save_config(config: Dict[str, Any], save_path: str):
    """
    Save configuration to YAML file.
    
    Args:
        config: Configuration dictionary
        save_path: Path to save configuration
    """
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(save_path, "w") as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)


def get_device(device_str: str = "auto") -> torch.device:
    """
    Get the appropriate device for training.
    
    Args:
        device_str: 'auto', 'cuda', 'cpu', or specific device like 'cuda:0'
        
    Returns:
        torch.device instance
    """
    if device_str == "auto":
        if torch.cuda.is_available():
            device = torch.device("cuda")
            print(f"Using GPU: {torch.cuda.get_device_name(0)}")
            print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        else:
            device = torch.device("cpu")
            print("CUDA not available, using CPU")
    else:
        device = torch.device(device_str)
        if device.type == "cuda":
            print(f"Using GPU: {torch.cuda.get_device_name(device.index or 0)}")
    
    return device


def set_seed(seed: int = 42):
    """
    Set random seeds for reproducibility.
    
    Args:
        seed: Random seed value
    """
    import random
    import numpy as np
    
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    
    print(f"Random seed set to {seed}")


def get_project_root() -> Path:
    """Get the project root directory."""
    current = Path(__file__).resolve()
    
    # Go up until we find the project root (contains configs folder)
    for parent in current.parents:
        if (parent / "configs").exists():
            return parent
    
    return current.parent.parent.parent


def create_output_dirs(base_dir: str = "outputs"):
    """Create output directories for training."""
    base_path = Path(base_dir)
    
    dirs = [
        base_path / "logs",
        base_path / "figures",
        base_path / "predictions",
        Path("checkpoints"),
    ]
    
    for d in dirs:
        d.mkdir(parents=True, exist_ok=True)
    
    print(f"Created output directories in {base_path}")


if __name__ == "__main__":
    # Test configuration utilities
    config_path = "configs/training_config.yaml"
    
    if os.path.exists(config_path):
        config = load_config(config_path)
        print("Loaded config:")
        print(f"  Model: {config.get('model', {}).get('name', 'N/A')}")
        print(f"  Batch size: {config.get('training', {}).get('batch_size', 'N/A')}")
        print(f"  Learning rate: {config.get('optimizer', {}).get('lr', 'N/A')}")
    
    device = get_device("auto")
    print(f"Device: {device}")
