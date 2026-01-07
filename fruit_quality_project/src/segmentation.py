"""
Segmentation module for fruit isolation.
Implements GrabCut and HSV-based segmentation methods.
"""

import cv2
import numpy as np
from typing import Tuple, Optional, Dict, Any


def segment_grabcut(
    image: np.ndarray,
    rect: Optional[Tuple[int, int, int, int]] = None,
    iterations: int = 5,
    margin: int = 10
) -> Dict[str, Any]:
    """
    Segment fruit using OpenCV GrabCut algorithm.
    
    Args:
        image: Input image (RGB format)
        rect: Rectangle (x, y, w, h) for initialization. Auto-computed if None.
        iterations: Number of GrabCut iterations
        margin: Margin for automatic rectangle computation
        
    Returns:
        Dictionary with:
            - 'mask': Binary segmentation mask
            - 'segmented': Segmented image (black background)
            - 'cropped': Cropped image to bounding box
            - 'bbox': Bounding box (x, y, w, h)
            - 'method_info': String describing method parameters
    """
    h, w = image.shape[:2]
    
    # Auto-compute rectangle if not provided
    if rect is None:
        rect = (margin, margin, w - 2 * margin, h - 2 * margin)
    
    # Initialize masks for GrabCut
    mask = np.zeros((h, w), np.uint8)
    bgd_model = np.zeros((1, 65), np.float64)
    fgd_model = np.zeros((1, 65), np.float64)
    
    # Convert to BGR for OpenCV
    img_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    
    # Apply GrabCut
    try:
        cv2.grabCut(img_bgr, mask, rect, bgd_model, fgd_model, 
                    iterations, cv2.GC_INIT_WITH_RECT)
    except cv2.error as e:
        print(f"[WARNING] GrabCut failed: {e}. Using fallback segmentation.")
        # Fallback: use center region
        mask = np.zeros((h, w), np.uint8)
        center_mask = np.zeros((h, w), np.uint8)
        cv2.ellipse(center_mask, (w//2, h//2), (w//3, h//3), 0, 0, 360, 1, -1)
        mask = center_mask
    
    # Create binary mask (foreground = 1, background = 0)
    binary_mask = np.where((mask == cv2.GC_FGD) | (mask == cv2.GC_PR_FGD), 255, 0).astype(np.uint8)
    
    # Apply mask to get segmented image
    segmented = image.copy()
    segmented[binary_mask == 0] = 0
    
    # Find bounding box from mask
    bbox = get_bounding_box(binary_mask)
    
    # Crop to bounding box
    if bbox is not None:
        x, y, bw, bh = bbox
        cropped = segmented[y:y+bh, x:x+bw]
    else:
        cropped = segmented
        bbox = (0, 0, w, h)
    
    method_info = f"GrabCut (iterations={iterations}, rect={rect})"
    
    return {
        'mask': binary_mask,
        'segmented': segmented,
        'cropped': cropped,
        'bbox': bbox,
        'method_info': method_info
    }


def segment_hsv(
    image: np.ndarray,
    lower_hsv: Optional[Tuple[int, int, int]] = None,
    upper_hsv: Optional[Tuple[int, int, int]] = None,
    kernel_size: int = 5,
    morph_iterations: int = 2
) -> Dict[str, Any]:
    """
    Segment fruit using HSV color thresholding and morphological operations.
    
    Args:
        image: Input image (RGB format)
        lower_hsv: Lower HSV bounds. Auto-detected if None.
        upper_hsv: Upper HSV bounds. Auto-detected if None.
        kernel_size: Size of morphological kernel
        morph_iterations: Number of morphological iterations
        
    Returns:
        Dictionary with same structure as segment_grabcut
    """
    h, w = image.shape[:2]
    
    # Convert to HSV
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    
    # Auto-detect color ranges if not provided
    if lower_hsv is None or upper_hsv is None:
        lower_hsv, upper_hsv = auto_detect_hsv_range(hsv)
    
    # Create mask from HSV thresholding
    mask1 = cv2.inRange(hsv, np.array(lower_hsv), np.array(upper_hsv))
    
    # For fruits like apples/oranges, we might need multiple ranges (red wraps around)
    # Add secondary range for red hues (0-10 and 160-180)
    if lower_hsv[0] < 20:  # Likely looking for red/orange
        lower_hsv2 = (160, lower_hsv[1], lower_hsv[2])
        upper_hsv2 = (180, upper_hsv[1], upper_hsv[2])
        mask2 = cv2.inRange(hsv, np.array(lower_hsv2), np.array(upper_hsv2))
        mask1 = cv2.bitwise_or(mask1, mask2)
    
    # Morphological operations to clean up
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    
    # Opening (remove noise)
    binary_mask = cv2.morphologyEx(mask1, cv2.MORPH_OPEN, kernel, iterations=morph_iterations)
    
    # Closing (fill holes)
    binary_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_CLOSE, kernel, iterations=morph_iterations)
    
    # Find largest contour (main fruit)
    contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if contours:
        # Keep only the largest contour
        largest_contour = max(contours, key=cv2.contourArea)
        binary_mask = np.zeros((h, w), np.uint8)
        cv2.drawContours(binary_mask, [largest_contour], -1, 255, -1)
    
    # Apply mask
    segmented = image.copy()
    segmented[binary_mask == 0] = 0
    
    # Find bounding box
    bbox = get_bounding_box(binary_mask)
    
    # Crop
    if bbox is not None:
        x, y, bw, bh = bbox
        cropped = segmented[y:y+bh, x:x+bw]
    else:
        cropped = segmented
        bbox = (0, 0, w, h)
    
    method_info = f"HSV + morphology (lower={lower_hsv}, upper={upper_hsv}, kernel={kernel_size})"
    
    return {
        'mask': binary_mask,
        'segmented': segmented,
        'cropped': cropped,
        'bbox': bbox,
        'method_info': method_info
    }


def auto_detect_hsv_range(hsv_image: np.ndarray) -> Tuple[Tuple, Tuple]:
    """
    Auto-detect HSV range for fruit segmentation.
    Uses center region statistics to estimate fruit color.
    
    Args:
        hsv_image: Image in HSV format
        
    Returns:
        Tuple of (lower_hsv, upper_hsv) bounds
    """
    h, w = hsv_image.shape[:2]
    
    # Sample from center region (likely the fruit)
    center_y, center_x = h // 2, w // 2
    margin_y, margin_x = h // 4, w // 4
    
    center_region = hsv_image[
        center_y - margin_y:center_y + margin_y,
        center_x - margin_x:center_x + margin_x
    ]
    
    # Compute statistics
    h_mean = np.mean(center_region[:, :, 0])
    s_mean = np.mean(center_region[:, :, 1])
    v_mean = np.mean(center_region[:, :, 2])
    
    # Define ranges based on statistics
    h_range = 25
    s_lower = max(40, s_mean - 60)
    v_lower = max(40, v_mean - 80)
    
    lower_hsv = (max(0, int(h_mean - h_range)), int(s_lower), int(v_lower))
    upper_hsv = (min(180, int(h_mean + h_range)), 255, 255)
    
    return lower_hsv, upper_hsv


def get_bounding_box(mask: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
    """
    Get bounding box from binary mask.
    
    Args:
        mask: Binary mask
        
    Returns:
        Tuple (x, y, width, height) or None if mask is empty
    """
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        return None
    
    # Find overall bounding box
    x_min, y_min = mask.shape[1], mask.shape[0]
    x_max, y_max = 0, 0
    
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        x_min = min(x_min, x)
        y_min = min(y_min, y)
        x_max = max(x_max, x + w)
        y_max = max(y_max, y + h)
    
    return (x_min, y_min, x_max - x_min, y_max - y_min)


def segment_image(
    image: np.ndarray,
    method: str = "grabcut",
    **kwargs
) -> Dict[str, Any]:
    """
    Unified segmentation interface.
    
    Args:
        image: Input image (RGB)
        method: 'grabcut' or 'hsv'
        **kwargs: Method-specific parameters
        
    Returns:
        Segmentation results dictionary
    """
    method = method.lower()
    
    if method == "grabcut":
        return segment_grabcut(image, **kwargs)
    elif method == "hsv":
        return segment_hsv(image, **kwargs)
    else:
        raise ValueError(f"Unknown segmentation method: {method}. Use 'grabcut' or 'hsv'.")


def apply_segmentation_to_dataset(
    image_paths: list,
    output_dir: str,
    method: str = "grabcut",
    max_samples: int = 5,
    **kwargs
) -> list:
    """
    Apply segmentation to multiple images and save samples.
    
    Args:
        image_paths: List of image paths
        output_dir: Directory to save segmented samples
        method: Segmentation method
        max_samples: Maximum number of samples to save
        **kwargs: Method-specific parameters
        
    Returns:
        List of paths to saved segmented images
    """
    import os
    from .utils import load_image, save_image
    
    os.makedirs(output_dir, exist_ok=True)
    saved_paths = []
    
    for i, path in enumerate(image_paths[:max_samples]):
        try:
            image = load_image(path)
            result = segment_image(image, method=method, **kwargs)
            
            # Save segmented image
            filename = os.path.basename(path)
            name, ext = os.path.splitext(filename)
            
            save_path = os.path.join(output_dir, f"{name}_segmented{ext}")
            save_image(result['segmented'], save_path)
            
            # Save mask
            mask_path = os.path.join(output_dir, f"{name}_mask{ext}")
            cv2.imwrite(mask_path, result['mask'])
            
            saved_paths.append(save_path)
            print(f"[INFO] Segmented: {filename} | {result['method_info']}")
            
        except Exception as e:
            print(f"[WARNING] Failed to segment {path}: {e}")
    
    return saved_paths
