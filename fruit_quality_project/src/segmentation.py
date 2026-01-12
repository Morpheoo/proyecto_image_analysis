import cv2
import numpy as np
from typing import Tuple, Optional, Dict, Any, List


def get_hsv_channels(image: np.ndarray) -> Dict[str, np.ndarray]:
    """
    Extract H, S, V channels separately for academic visualization.
    """
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    return {
        'H': hsv[:, :, 0],
        'S': hsv[:, :, 1],
        'V': hsv[:, :, 2]
    }


def apply_filters(image: np.ndarray, method: str = "gaussian", kernel_size: int = 5) -> np.ndarray:
    """
    Apply image filtering for noise reduction.
    """
    if method == "gaussian":
        return cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)
    elif method == "median":
        return cv2.medianBlur(image, kernel_size)
    return image


def segment_grabcut(
    image: np.ndarray,
    rect: Optional[Tuple[int, int, int, int]] = None,
    iterations: int = 5,
    margin: int = 10,
    **kwargs
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
    morph_iterations: int = 2,
    **kwargs
) -> Dict[str, Any]:
    """
    Segmentación robusta usando combinación de Otsu (forma) y Rangos HSV (color).
    """
    h, w = image.shape[:2]
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    
    # 1. Máscara de Otsu sobre el canal de Saturación (Excelente para fondos blancos/claros)
    s_channel = hsv[:, :, 1]
    _, otsu_mask = cv2.threshold(s_channel, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # 2. Máscara de Rangos de Color
    if lower_hsv is None or upper_hsv is None:
        lower_hsv, upper_hsv = auto_detect_hsv_range(hsv)
    
    # Manejar el wrap-around del Rojo (0-180)
    if lower_hsv[0] > upper_hsv[0]:
        mask1 = cv2.inRange(hsv, np.array((lower_hsv[0], lower_hsv[1], lower_hsv[2])), np.array((180, upper_hsv[1], upper_hsv[2])))
        mask2 = cv2.inRange(hsv, np.array((0, lower_hsv[1], lower_hsv[2])), np.array((upper_hsv[0], upper_hsv[1], upper_hsv[2])))
        range_mask = cv2.bitwise_or(mask1, mask2)
    else:
        range_mask = cv2.inRange(hsv, np.array(lower_hsv), np.array(upper_hsv))
    
    # 3. Combinación Adaptativa
    # Si detectamos un fondo blanco/claro, la máscara de Otsu es sumamente confiable.
    # El AND con range_mask puede ser demasiado agresivo.
    binary_mask = cv2.bitwise_and(otsu_mask, range_mask)
    
    # Salvaguarda: Si la combinación borró casi todo el objeto detectado por Otsu, 
    # confiamos más en la forma (Otsu) que en el color detectado.
    if np.sum(binary_mask > 0) < 0.2 * np.sum(otsu_mask > 0):
        binary_mask = otsu_mask
    
    # Limpieza Morfológica
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    binary_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_OPEN, kernel, iterations=morph_iterations)
    binary_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_CLOSE, kernel, iterations=morph_iterations)

    # Aislamiento del Contorno más grande
    contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        largest_contour = max(contours, key=cv2.contourArea)
        binary_mask = np.zeros((h, w), np.uint8)
        cv2.drawContours(binary_mask, [largest_contour], -1, 255, -1)
    
    # Generar resultado segmentado
    segmented = image.copy()
    segmented[binary_mask == 0] = 0
    bbox = get_bounding_box(binary_mask)
    
    if bbox is not None:
        x, y, bw, bh = bbox
        cropped = segmented[y:y+bh, x:x+bw]
    else:
        cropped = segmented
        bbox = (0, 0, w, h)
    
    method_info = f"HSV Adaptativo (lower={lower_hsv}, upper={upper_hsv})"
    
    return {
        'mask': binary_mask,
        'segmented': segmented,
        'cropped': cropped,
        'bbox': bbox,
        'method_info': method_info
    }


def auto_detect_hsv_range(hsv_image: np.ndarray) -> Tuple[Tuple, Tuple]:
    """
    Detección automática de rango filtrando píxeles de poco interés (fondos).
    """
    # Tomar región central para muestreo
    h, w = hsv_image.shape[:2]
    cy, cx = h // 2, w // 2
    my, mx = h // 6, w // 6
    roi = hsv_image[cy-my:cy+my, cx-mx:cx+mx]
    
    # Filtrar solo píxeles con saturación y brillo decente (ignorar blancos/negros)
    valid_pixels = roi[(roi[:, :, 1] > 30) & (roi[:, :, 2] > 30)]
    
    if len(valid_pixels) > 0:
        h_vals = valid_pixels[:, 0]
        s_vals = valid_pixels[:, 1]
        v_vals = valid_pixels[:, 2]
        
        h_mean = np.mean(h_vals)
        s_mean = np.mean(s_vals)
        v_mean = np.mean(v_vals)
        
        # Margen dinámico
        h_min = (h_mean - 20) % 180
        h_max = (h_mean + 20) % 180
        lower = (int(h_min), max(40, int(s_mean*0.6)), max(40, int(v_mean*0.4)))
        upper = (int(h_max), 255, 255)
    else:
        # Fallback si el centro es vacío/blanco: Rango amplio
        lower = (0, 40, 40)
        upper = (180, 255, 255)
        
    return lower, upper


def get_solidity(contour) -> float:
    """
    Calculate solidity: Ratio of contour area to its convex hull area.
    """
    area = cv2.contourArea(contour)
    hull = cv2.convexHull(contour)
    hull_area = cv2.contourArea(hull)
    if hull_area > 0:
        return float(area) / hull_area
    return 0


def get_edges(image: np.ndarray, low_threshold: int = 50, high_threshold: int = 150) -> np.ndarray:
    """
    Apply Canny edge detection (Discontinuity segmentation).
    """
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(gray, low_threshold, high_threshold)
    return edges


def get_bounding_box(mask: np.ndarray, padding: int = 10) -> Optional[Tuple[int, int, int, int]]:
    """
    Get bounding box from binary mask with optional padding.
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
    
    # Apply padding and ensure it's within image bounds
    h, w = mask.shape
    x_min = max(0, x_min - padding)
    y_min = max(0, y_min - padding)
    x_max = min(w, x_max + padding)
    y_max = min(h, y_max + padding)
    
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
