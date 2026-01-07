"""
Utility functions for the Fruit Quality Classification project.
Provides reproducibility, logging, and I/O helpers.
"""

import os
import random
import logging
from datetime import datetime

import numpy as np
import torch
from PIL import Image
import cv2


def set_seed(seed: int = 42) -> None:
    """
    Set random seed for reproducibility across all libraries.
    
    Args:
        seed: Random seed value
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    print(f"[INFO] Seed set to {seed} for reproducibility")


def setup_logging(log_file: str = None) -> logging.Logger:
    """
    Configure logging for the project.
    
    Args:
        log_file: Optional path to log file
        
    Returns:
        Configured logger instance
    """
    logger = logging.getLogger("FruitQuality")
    logger.setLevel(logging.INFO)
    
    formatter = logging.Formatter(
        '[%(asctime)s] [%(levelname)s] %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # File handler (optional)
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger


def create_directories(base_path: str) -> None:
    """
    Create all required project directories.
    
    Args:
        base_path: Base path of the project
    """
    directories = [
        os.path.join(base_path, "data"),
        os.path.join(base_path, "models"),
        os.path.join(base_path, "outputs", "segmentation_samples"),
        os.path.join(base_path, "outputs", "predictions_samples"),
        os.path.join(base_path, "outputs", "streamlit_samples"),
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
    
    print(f"[INFO] Created project directories in {base_path}")


def load_image(path: str, mode: str = "RGB") -> np.ndarray:
    """
    Load an image from disk.
    
    Args:
        path: Path to the image file
        mode: Color mode ('RGB' or 'BGR')
        
    Returns:
        Image as numpy array
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Image not found: {path}")
    
    img = cv2.imread(path)
    if img is None:
        raise ValueError(f"Could not read image: {path}")
    
    if mode == "RGB":
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    return img


def save_image(image: np.ndarray, path: str, mode: str = "RGB") -> None:
    """
    Save an image to disk.
    
    Args:
        image: Image as numpy array
        path: Destination path
        mode: Color mode of input ('RGB' or 'BGR')
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    
    if mode == "RGB":
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    
    cv2.imwrite(path, image)


def get_device() -> torch.device:
    """
    Get the best available device for PyTorch.
    
    Returns:
        torch.device (cuda if available, else cpu)
    """
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"[INFO] Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device("cpu")
        print("[INFO] Using CPU")
    
    return device


def get_timestamp() -> str:
    """
    Get current timestamp as string.
    
    Returns:
        Timestamp string
    """
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def count_parameters(model: torch.nn.Module) -> int:
    """
    Count trainable parameters in a model.
    
    Args:
        model: PyTorch model
        
    Returns:
        Number of trainable parameters
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def print_section(title: str, char: str = "=", width: int = 60) -> None:
    """
    Print a formatted section header.
    
    Args:
        title: Section title
        char: Character for the separator line
        width: Width of the separator
    """
    print(f"\n{char * width}")
    print(f" {title}")
    print(f"{char * width}\n")
