"""
Inference module for fruit quality classification.
Standalone module for loading model and running predictions.
"""

import argparse
import sys
from pathlib import Path
from typing import Optional, Dict, Any, Tuple

import numpy as np
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image

# Add project root to path for imports
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import (
    DEFAULT_MODEL_PATH, CLASS_NAMES, NUM_CLASSES,
    IMAGENET_MEAN, IMAGENET_STD, INPUT_SIZE, MODEL_ARCHITECTURE
)
from src.segmentation import segment_image


def get_device() -> torch.device:
    """Get the best available device (CUDA > CPU)."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def get_inference_transforms() -> transforms.Compose:
    """
    Get transforms for inference.
    Matches MobileNetV2 requirements: 224x224, ImageNet normalization.
    """
    return transforms.Compose([
        transforms.Resize((INPUT_SIZE, INPUT_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
    ])


def create_model(
    model_name: str = MODEL_ARCHITECTURE,
    num_classes: int = NUM_CLASSES
) -> nn.Module:
    """
    Create a MobileNetV2 model structure for loading weights.
    
    Args:
        model_name: Model architecture name
        num_classes: Number of output classes
        
    Returns:
        PyTorch model (without pretrained weights)
    """
    if model_name == "mobilenetv2":
        model = models.mobilenet_v2(weights=None)
        in_features = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(in_features, num_classes)
    elif model_name == "resnet18":
        model = models.resnet18(weights=None)
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, num_classes)
    else:
        raise ValueError(f"Unknown model architecture: {model_name}")
    
    return model


def load_inference_model(
    model_path: Optional[str] = None,
    device: Optional[torch.device] = None
) -> Tuple[nn.Module, torch.device, Dict[str, Any]]:
    """
    Load trained model for inference.
    
    Args:
        model_path: Path to model checkpoint (.pth file)
        device: Device to load model on (auto-detected if None)
        
    Returns:
        Tuple of (model, device, checkpoint_info)
        
    Raises:
        FileNotFoundError: If model file doesn't exist
        RuntimeError: If model loading fails
    """
    if model_path is None:
        model_path = str(DEFAULT_MODEL_PATH)
    
    if device is None:
        device = get_device()
    
    path = Path(model_path)
    if not path.exists():
        raise FileNotFoundError(
            f"Model not found at: {model_path}\n"
            f"Expected location: {DEFAULT_MODEL_PATH}"
        )
    
    # Load checkpoint
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    
    # Handle both full checkpoint and state_dict only
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
        info = {
            'epoch': checkpoint.get('epoch', 'N/A'),
            'val_acc': checkpoint.get('val_acc', 'N/A'),
            'path': str(model_path)
        }
    else:
        # Assume it's just the state_dict
        state_dict = checkpoint
        info = {'path': str(model_path), 'epoch': 'N/A', 'val_acc': 'N/A'}
    
    # Create model and load weights
    model = create_model()
    model.load_state_dict(state_dict)
    model = model.to(device)
    model.eval()
    
    return model, device, info


def predict(
    model: nn.Module,
    image: np.ndarray,
    device: torch.device,
    preprocess: Optional[str] = None
) -> Dict[str, Any]:
    """
    Run inference on a single image.
    
    Args:
        model: Loaded PyTorch model
        image: Input image as numpy array (RGB)
        device: Device to run inference on
        preprocess: Segmentation method ('none', 'grabcut', 'hsv') or None
        
    Returns:
        Dictionary with:
            - prediction: Class name ('Fresh' or 'Rotten')
            - label_idx: Class index
            - confidence: Prediction confidence (0-1)
            - probabilities: Dict of class probabilities
            - processed_image: Image used for inference (after preprocessing)
            - mask: Segmentation mask (if applicable)
    """
    transform = get_inference_transforms()
    processed_image = image.copy()
    mask = None
    seg_info = "None (original image)"
    
    # Apply segmentation if requested
    if preprocess and preprocess.lower() not in ['none', 'original', '']:
        try:
            seg_result = segment_image(image, method=preprocess.lower())
            processed_image = seg_result['segmented']
            mask = seg_result['mask']
            seg_info = seg_result.get('method_info', preprocess)
        except Exception as e:
            print(f"[WARNING] Segmentation failed: {e}. Using original image.")
    
    # Convert to PIL and apply transforms
    pil_image = Image.fromarray(processed_image)
    img_tensor = transform(pil_image).unsqueeze(0).to(device)
    
    # Inference
    with torch.no_grad():
        outputs = model(img_tensor)
        probs = torch.nn.functional.softmax(outputs, dim=1)
        _, pred = outputs.max(1)
    
    pred_label = pred.item()
    confidence = probs[0][pred_label].item()
    
    return {
        'prediction': CLASS_NAMES[pred_label],
        'label_idx': pred_label,
        'confidence': confidence,
        'probabilities': {
            CLASS_NAMES[i]: float(probs[0][i]) 
            for i in range(len(CLASS_NAMES))
        },
        'processed_image': processed_image,
        'mask': mask,
        'preprocessing': seg_info
    }


def check_model_status() -> Dict[str, Any]:
    """
    Check if model is available and loadable.
    
    Returns:
        Dictionary with status information
    """
    status = {
        'model_exists': False,
        'model_loadable': False,
        'model_path': str(DEFAULT_MODEL_PATH),
        'error': None,
        'info': None
    }
    
    if DEFAULT_MODEL_PATH.exists():
        status['model_exists'] = True
        try:
            _, device, info = load_inference_model()
            status['model_loadable'] = True
            status['device'] = str(device)
            status['info'] = info
        except Exception as e:
            status['error'] = str(e)
    else:
        status['error'] = f"Model file not found at {DEFAULT_MODEL_PATH}"
    
    return status


def main():
    """CLI for running inference on a single image."""
    parser = argparse.ArgumentParser(
        description="Fruit Quality Classification - Inference"
    )
    parser.add_argument(
        "--image", "-i",
        type=str,
        required=True,
        help="Path to input image"
    )
    parser.add_argument(
        "--preprocess", "-p",
        type=str,
        choices=['none', 'grabcut', 'hsv'],
        default='none',
        help="Preprocessing/segmentation method"
    )
    parser.add_argument(
        "--model", "-m",
        type=str,
        default=None,
        help="Path to model checkpoint (optional)"
    )
    
    args = parser.parse_args()
    
    # Check image exists
    img_path = Path(args.image)
    if not img_path.exists():
        print(f"[ERROR] Image not found: {args.image}")
        sys.exit(1)
    
    # Load model
    print(f"[INFO] Loading model...")
    try:
        model, device, info = load_inference_model(args.model)
        print(f"[INFO] Model loaded successfully on {device}")
        if info.get('val_acc') != 'N/A':
            print(f"[INFO] Model validation accuracy: {info['val_acc']:.2f}%")
    except Exception as e:
        print(f"[ERROR] Failed to load model: {e}")
        sys.exit(1)
    
    # Load and process image
    print(f"[INFO] Processing image: {args.image}")
    image = np.array(Image.open(args.image).convert('RGB'))
    
    # Run inference
    print(f"[INFO] Preprocessing: {args.preprocess}")
    result = predict(model, image, device, args.preprocess)
    
    # Print results
    print("\n" + "="*50)
    print("PREDICTION RESULTS")
    print("="*50)
    print(f"  Prediction: {result['prediction']}")
    print(f"  Confidence: {result['confidence']*100:.1f}%")
    print(f"  Fresh probability: {result['probabilities']['Fresh']*100:.1f}%")
    print(f"  Rotten probability: {result['probabilities']['Rotten']*100:.1f}%")
    print(f"  Preprocessing: {result['preprocessing']}")
    print("="*50)


if __name__ == "__main__":
    main()
