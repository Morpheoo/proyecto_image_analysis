"""
Configuration module for fruit quality classification.
Contains model paths, class names, and inference parameters.
"""

from pathlib import Path

# Project paths
PROJECT_ROOT = Path(__file__).parent.parent
MODELS_DIR = PROJECT_ROOT / "models"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
STREAMLIT_SAMPLES_DIR = OUTPUTS_DIR / "streamlit_samples"

# Model configuration
DEFAULT_MODEL_NAME = "fruit_quality_baseline.pth"
DEFAULT_MODEL_PATH = MODELS_DIR / DEFAULT_MODEL_NAME

# Class configuration
# Binary classification: Fresh (0) vs Rotten (1)
CLASS_NAMES = ['Fresh', 'Rotten']
NUM_CLASSES = 2

# If model was trained with 6 classes (fruit-specific)
EXTENDED_CLASS_NAMES = [
    'freshapples', 'freshbanana', 'freshoranges',
    'rottenapples', 'rottenbanana', 'rottenoranges'
]

# ImageNet normalization (used by MobileNetV2)
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# Image preprocessing
INPUT_SIZE = 224

# Model architecture
MODEL_ARCHITECTURE = "mobilenetv2"

# Segmentation methods
SEGMENTATION_METHODS = {
    "none": "No preprocessing - use original image",
    "grabcut": "GrabCut (OpenCV) - iterative graph-based segmentation",
    "hsv": "HSV + Morphology - color thresholding with morphological ops"
}
