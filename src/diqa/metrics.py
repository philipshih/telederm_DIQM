"""
Core module for calculating Dermatologic Image Quality Metrics (D-IQMs).
"""

from pathlib import Path
import cv2
import numpy as np

def compute_sharpness_laplacian(image_path: Path) -> float:
    """
    Computes the sharpness of an image using the variance of the Laplacian.
    A higher value indicates a sharper image.

    Args:
        image_path: Path to the image file.

    Returns:
        The variance of the Laplacian, a float indicating sharpness.
    """
    image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        return 0.0
    laplacian_var = cv2.Laplacian(image, cv2.CV_64F).var()
    return laplacian_var

def compute_exposure_histogram(image_path: Path) -> tuple[float, float]:
    """
    Computes the percentage of under- and over-exposed pixels in an image.

    Args:
        image_path: Path to the image file.

    Returns:
        A tuple containing the percentage of under-exposed and over-exposed pixels.
    """
    image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        return 0.0, 0.0
    
    hist = cv2.calcHist([image], [0], None, [256], [0, 256])
    total_pixels = image.shape[0] * image.shape[1]
    
    under_exposed_pct = hist[:10].sum() / total_pixels
    over_exposed_pct = hist[246:].sum() / total_pixels
    
    return float(under_exposed_pct), float(over_exposed_pct)

def compute_glare_hsv(image_path: Path) -> float:
    """
    Computes the percentage of pixels that are likely specular glare.
    Uses thresholding in the HSV color space.

    Args:
        image_path: Path to the image file.

    Returns:
        The percentage of glare pixels in the image.
    """
    image = cv2.imread(str(image_path))
    if image is None:
        return 0.0
        
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    _, s, v = cv2.split(hsv)
    
    # Glare is typically low saturation and high value (brightness)
    glare_mask = cv2.inRange(s, 0, 30) & cv2.inRange(v, 220, 255)
    
    total_pixels = image.shape[0] * image.shape[1]
    glare_pct = np.sum(glare_mask > 0) / total_pixels
    
    return float(glare_pct)
