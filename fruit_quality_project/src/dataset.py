"""
Dataset module for fruit quality classification.
Handles data loading, preprocessing, and augmentation.
"""

import os
from typing import Tuple, List, Dict, Optional, Callable
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import cv2

from .segmentation import segment_image


# Class mappings
CLASS_NAMES = ['freshapples', 'freshbanana', 'freshoranges', 
               'rottenapples', 'rottenbanana', 'rottenoranges']

# Binary classification: fresh (0) vs rotten (1)
QUALITY_LABELS = {
    'freshapples': 0, 'freshbanana': 0, 'freshoranges': 0,
    'rottenapples': 1, 'rottenbanana': 1, 'rottenoranges': 1
}

QUALITY_NAMES = ['Fresh', 'Rotten']


class FruitDataset(Dataset):
    """
    PyTorch Dataset for fruit quality classification.
    Supports both original and segmented images.
    """
    
    def __init__(
        self,
        root_dir: str,
        split: str = "train",
        transform: Optional[Callable] = None,
        use_segmentation: bool = False,
        segmentation_method: str = "grabcut",
        cache_segmentation: bool = True
    ):
        """
        Initialize the dataset.
        
        Args:
            root_dir: Path to dataset root (contains train/ and test/ folders)
            split: 'train' or 'test'
            transform: Torchvision transforms to apply
            use_segmentation: Whether to apply segmentation before classification
            segmentation_method: 'grabcut' or 'hsv'
            cache_segmentation: Whether to cache segmented images
        """
        self.root_dir = Path(root_dir)
        self.split = split
        self.transform = transform
        self.use_segmentation = use_segmentation
        self.segmentation_method = segmentation_method
        self.cache_segmentation = cache_segmentation
        
        self.split_dir = self.root_dir / split
        
        if not self.split_dir.exists():
            raise FileNotFoundError(
                f"Dataset split directory not found: {self.split_dir}\n"
                f"Please download the dataset and place it in {self.root_dir}"
            )
        
        # Collect all image paths and labels
        self.samples: List[Tuple[str, int]] = []
        self._load_samples()
        
        # Cache for segmented images
        self._seg_cache: Dict[str, np.ndarray] = {}
        
        print(f"[INFO] Loaded {len(self.samples)} images from {split} split")
        print(f"[INFO] Segmentation: {'Enabled - ' + segmentation_method if use_segmentation else 'Disabled'}")
    
    def _load_samples(self):
        """Load all image paths and their labels."""
        for class_name in CLASS_NAMES:
            class_dir = self.split_dir / class_name
            
            if not class_dir.exists():
                print(f"[WARNING] Class directory not found: {class_dir}")
                continue
            
            label = QUALITY_LABELS[class_name]
            
            for img_file in class_dir.iterdir():
                if img_file.suffix.lower() in ['.jpg', '.jpeg', '.png']:
                    self.samples.append((str(img_file), label))
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        img_path, label = self.samples[idx]
        
        # Load image
        image = Image.open(img_path).convert('RGB')
        image_np = np.array(image)
        
        # Apply segmentation if enabled
        if self.use_segmentation:
            if self.cache_segmentation and img_path in self._seg_cache:
                image_np = self._seg_cache[img_path]
            else:
                try:
                    result = segment_image(image_np, method=self.segmentation_method)
                    image_np = result['cropped']
                    
                    # Ensure minimum size
                    if image_np.shape[0] < 10 or image_np.shape[1] < 10:
                        image_np = np.array(image)  # Fallback to original
                    
                    if self.cache_segmentation:
                        self._seg_cache[img_path] = image_np
                        
                except Exception as e:
                    # Fallback to original image on segmentation failure
                    pass
            
            image = Image.fromarray(image_np)
        
        # Apply transforms
        if self.transform:
            image = self.transform(image)
        
        return image, label
    
    def get_class_distribution(self) -> Dict[str, int]:
        """Get distribution of classes in the dataset."""
        distribution = {name: 0 for name in QUALITY_NAMES}
        
        for _, label in self.samples:
            distribution[QUALITY_NAMES[label]] += 1
        
        return distribution


def get_transforms(split: str = "train", img_size: int = 224) -> transforms.Compose:
    """
    Get image transforms for training or evaluation.
    
    Args:
        split: 'train' or 'test'
        img_size: Target image size
        
    Returns:
        Composed transforms
    """
    # ImageNet normalization
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
    
    if split == "train":
        return transforms.Compose([
            transforms.Resize((img_size + 32, img_size + 32)),
            transforms.RandomCrop(img_size),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(15),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.ToTensor(),
            normalize
        ])
    else:
        return transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            normalize
        ])


def get_dataloaders(
    data_dir: str,
    batch_size: int = 32,
    num_workers: int = 4,
    use_segmentation: bool = False,
    segmentation_method: str = "grabcut",
    img_size: int = 224
) -> Tuple[DataLoader, DataLoader]:
    """
    Create train and test dataloaders.
    
    Args:
        data_dir: Path to dataset root
        batch_size: Batch size
        num_workers: Number of data loading workers
        use_segmentation: Whether to use segmentation
        segmentation_method: Segmentation method to use
        img_size: Target image size
        
    Returns:
        Tuple of (train_loader, test_loader)
    """
    train_transform = get_transforms("train", img_size)
    test_transform = get_transforms("test", img_size)
    
    train_dataset = FruitDataset(
        root_dir=data_dir,
        split="train",
        transform=train_transform,
        use_segmentation=use_segmentation,
        segmentation_method=segmentation_method
    )
    
    test_dataset = FruitDataset(
        root_dir=data_dir,
        split="test",
        transform=test_transform,
        use_segmentation=use_segmentation,
        segmentation_method=segmentation_method
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return train_loader, test_loader


def get_sample_images(data_dir: str, samples_per_class: int = 5) -> Dict[str, List[str]]:
    """
    Get sample image paths from each class.
    
    Args:
        data_dir: Path to dataset root
        samples_per_class: Number of samples per class
        
    Returns:
        Dictionary mapping class names to image paths
    """
    samples = {}
    data_path = Path(data_dir)
    
    for split in ["train", "test"]:
        split_dir = data_path / split
        if not split_dir.exists():
            continue
            
        for class_name in CLASS_NAMES:
            class_dir = split_dir / class_name
            if not class_dir.exists():
                continue
            
            if class_name not in samples:
                samples[class_name] = []
            
            img_files = list(class_dir.glob("*.jpg")) + list(class_dir.glob("*.png"))
            samples[class_name].extend([str(f) for f in img_files[:samples_per_class]])
    
    return samples
