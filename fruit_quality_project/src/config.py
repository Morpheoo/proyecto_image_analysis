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

# Fruit-specific HSV color ranges for Academic Recognition
# Format: {fruit_name: [(lower, upper), ...]} (List because red wraps around)
FRUIT_COLOR_RANGES = {
    "Manzana 🍎": [
        ((0, 20, 20), (10, 255, 255)),    # Rojo 1 (Sat bajada de 40 a 20)
        ((160, 20, 20), (180, 255, 255)), # Rojo 2
        ((35, 20, 20), (85, 255, 255))    # Verde
    ],
    "Banana 🍌": [
        ((20, 20, 20), (38, 255, 255))    # Amarillo/Marrón
    ],
    "Naranja 🍊": [
        ((10, 20, 20), (28, 255, 255))    # Naranja/Marrón (Hue extendido a 28)
    ]
}
