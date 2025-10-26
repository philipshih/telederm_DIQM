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

def compute_glare_adaptive(image_path: Path) -> float:
    """
    Computes the percentage of pixels that are likely specular glare.
    Uses adaptive, percentile-based thresholding in the HSV color space.

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

    # Adaptive thresholding: top 5% brightness, bottom 20% saturation
    v_threshold = np.percentile(v, 95)
    s_threshold = np.percentile(s, 20)

    glare_mask = (s < s_threshold) & (v > v_threshold)
    glare_pct = float(np.sum(glare_mask) / glare_mask.size)

    return glare_pct

def compute_contrast(image_path: Path) -> tuple[float, float]:
    """
    Computes global and local contrast metrics for the image.

    Contrast is critical for dermatology as it affects lesion boundary visibility
    and diagnostic utility. This metric is standard in all teledermatology IQA systems.

    Args:
        image_path: Path to the image file.

    Returns:
        A tuple containing:
        - global_contrast: RMS contrast (std of intensity), range varies by image
        - local_contrast: Michelson contrast in 32x32 patches, 0-1 scale
    """
    image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        return 0.0, 0.0

    # Global contrast: RMS (root mean square) contrast = standard deviation
    global_contrast = float(np.std(image))

    # Local contrast: Michelson contrast computed over patches
    # Michelson = (max - min) / (max + min)
    patch_size = 32
    h, w = image.shape
    local_contrasts = []

    for i in range(0, h - patch_size, patch_size):
        for j in range(0, w - patch_size, patch_size):
            patch = image[i:i+patch_size, j:j+patch_size]
            patch_max = patch.max()
            patch_min = patch.min()

            if patch_max + patch_min > 0:
                michelson = (patch_max - patch_min) / (patch_max + patch_min)
                local_contrasts.append(michelson)

    local_contrast = float(np.mean(local_contrasts)) if local_contrasts else 0.0

    return global_contrast, local_contrast


def compute_color_metrics(image_path: Path) -> tuple[float, float]:
    """
    Computes color fidelity metrics using the perceptually uniform LAB color space.

    Args:
        image_path: Path to the image file.

    Returns:
        A tuple containing:
        - color_variance: Variance in the chromatic 'a' and 'b' channels.
        - color_cast: Deviation from a neutral gray color balance.
    """
    image = cv2.imread(str(image_path))
    if image is None:
        return 0.0, 0.0

    # Convert to LAB (perceptually uniform)
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    _, a, b = cv2.split(lab)

    # Color variance in perceptual space (chromatic channels)
    color_variance = float(np.var(a) + np.var(b))

    # Color cast: deviation from neutral (a=128, b=128)
    a_mean = np.mean(a)
    b_mean = np.mean(b)
    color_cast = float(np.sqrt((a_mean - 128)**2 + (b_mean - 128)**2) / 128.0)

    return color_variance, color_cast


def compute_noise_level(image_path: Path) -> float:
    """
    Estimates noise level in the image using smooth region analysis.

    Noise is particularly relevant for smartphone teledermatology images.
    Common in consumer-grade cameras, especially in low light conditions.

    Args:
        image_path: Path to the image file.

    Returns:
        Estimated noise level (std in smooth regions), 0-∞ scale
    """
    image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        return 0.0

    # Apply median filter to get smooth version
    smooth = cv2.medianBlur(image, 5)

    # Noise estimate: std of difference between original and smoothed
    noise_map = cv2.absdiff(image, smooth)

    # Focus on regions with low gradient (smooth regions) for better estimate
    # Compute Laplacian to find low-texture regions
    laplacian = cv2.Laplacian(image, cv2.CV_64F)
    low_texture_mask = np.abs(laplacian) < np.percentile(np.abs(laplacian), 50)

    # Noise level: std in smooth regions
    if np.sum(low_texture_mask) > 0:
        noise_level = float(np.std(noise_map[low_texture_mask]))
    else:
        noise_level = float(np.std(noise_map))

    return noise_level


def compute_entropy(image_path: Path) -> float:
    """
    Computes image entropy as a measure of information content.

    Higher entropy indicates more information/complexity in the image.
    Low entropy may indicate uniform/featureless images lacking diagnostic value.
    FetMRQC found entropy to be an important feature.

    Args:
        image_path: Path to the image file.

    Returns:
        Shannon entropy of the intensity histogram (bits)
    """
    image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        return 0.0

    # Compute histogram
    hist, _ = np.histogram(image.flatten(), bins=256, range=(0, 256))

    # Normalize to probability distribution
    hist = hist / hist.sum()

    # Remove zero entries to avoid log(0)
    hist = hist[hist > 0]

    # Shannon entropy: -sum(p * log2(p))
    entropy = float(-np.sum(hist * np.log2(hist)))

    return entropy


def compute_edge_density(image_path: Path) -> float:
    """
    Computes edge density as an alternative sharpness/detail metric.

    Complements Laplacian sharpness by measuring proportion of edge pixels.
    Relevant for assessing if lesion boundaries are visible.

    Args:
        image_path: Path to the image file.

    Returns:
        Proportion of edge pixels (0-1 scale)
    """
    image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        return 0.0

    # Sobel edge detection
    sobel_x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)

    # Edge magnitude
    edge_magnitude = np.sqrt(sobel_x**2 + sobel_y**2)

    # Threshold to classify as edge (adaptive: top 20% of magnitudes)
    threshold = np.percentile(edge_magnitude, 80)
    edge_mask = edge_magnitude > threshold

    # Edge density: proportion of pixels classified as edges
    edge_density = float(np.sum(edge_mask) / edge_mask.size)

    return edge_density


def compute_dynamic_range(image_path: Path) -> float:
    """
    Computes dynamic range utilization of the image.

    Measures how well the image uses the available intensity range.
    Poor utilization indicates exposure problems complementing under/over metrics.

    Args:
        image_path: Path to the image file.

    Returns:
        Dynamic range utilization (0-1 scale, 1=full range used)
    """
    image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        return 0.0

    # Use 1st and 99th percentile to avoid outliers
    p1 = np.percentile(image, 1)
    p99 = np.percentile(image, 99)

    # Dynamic range utilization: (p99 - p1) / 255
    dynamic_range = float((p99 - p1) / 255.0)

    return dynamic_range


def compute_brisque_features(image_path: Path) -> tuple[float, float]:
    """
    Computes features inspired by the BRISQUE model, which are powerful
    indicators of image "naturalness" and texture quality.

    Args:
        image_path: Path to the image file.

    Returns:
        A tuple containing:
        - mscn_variance: Variance of the MSCN coefficients.
        - mscn_skewness: Skewness of the MSCN coefficients.
    """
    image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        return 0.0, 0.0

    # Compute Mean Subtracted Contrast Normalized (MSCN) coefficients
    # This is the core of BRISQUE's feature extraction
    mu = cv2.GaussianBlur(image, (7, 7), 1.166)
    mu_sq = mu * mu
    sigma = np.sqrt(abs(cv2.GaussianBlur(image * image, (7, 7), 1.166) - mu_sq))
    mscn = (image - mu) / (sigma + 1.0)

    # Return variance and skewness of the MSCN coefficients
    mscn_variance = float(np.var(mscn))
    if not np.isfinite(mscn_variance):
        mscn_variance = 0.0

    # Calculate skewness manually to avoid dependency on scipy
    n = mscn.size
    m2 = np.sum((mscn - np.mean(mscn))**2)
    m3 = np.sum((mscn - np.mean(mscn))**3)
    
    if m2 == 0:
        mscn_skewness = 0.0
    else:
        with np.errstate(divide='ignore', invalid='ignore'):
            mscn_skewness = (m3 / n) / (m2 / n)**1.5
        if not np.isfinite(mscn_skewness):
            mscn_skewness = 0.0

    return mscn_variance, float(mscn_skewness)


def compute_lesion_framing(image_path: Path) -> tuple[float, float]:
    """
    Computes framing metrics for the primary lesion using saliency detection.

    This is a segmentation-free approach that estimates whether the lesion
    is appropriately sized and centered in the frame.

    Args:
        image_path: Path to the image file.

    Returns:
        A tuple containing:
        - lesion_size_ratio: Proportion of image occupied by salient region (0.0-1.0)
        - lesion_centrality: How centered the lesion is (0.0-1.0, 1.0=perfectly centered)
    """
    image = cv2.imread(str(image_path))
    if image is None:
        return 0.0, 0.0

    h, w = image.shape[:2]
    total_pixels = h * w

    try:
        # Use static saliency detection to find most prominent region
        saliency = cv2.saliency.StaticSaliencyFineGrained_create()
        success, saliency_map = saliency.computeSaliency(image)

        if not success or saliency_map is None:
            # Fallback: use edge density in center
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            edges = cv2.Canny(gray, 50, 150)
            center_region = edges[h//4:3*h//4, w//4:3*w//4]
            lesion_size_ratio = center_region.sum() / (edges.sum() + 1e-6)
            lesion_centrality = 0.75  # Assume reasonably centered if using center crop
            return float(lesion_size_ratio), float(lesion_centrality)

        # Threshold saliency map to find prominent region
        # Use adaptive threshold: high saliency regions (top 20% of saliency values)
        threshold = np.percentile(saliency_map, 80)
        salient_mask = saliency_map > threshold

        # Lesion size ratio: what proportion of image is the salient region
        salient_pixels = np.sum(salient_mask)
        lesion_size_ratio = salient_pixels / total_pixels

        # Lesion centrality: how close is the centroid to image center
        if salient_pixels > 0:
            # Find centroid of salient region
            y_coords, x_coords = np.where(salient_mask)
            centroid_y = np.mean(y_coords)
            centroid_x = np.mean(x_coords)

            # Calculate normalized distance from center
            center_y, center_x = h / 2, w / 2
            distance_y = abs(centroid_y - center_y) / (h / 2)
            distance_x = abs(centroid_x - center_x) / (w / 2)
            euclidean_distance = np.sqrt(distance_x**2 + distance_y**2)

            # Convert to centrality score (1.0 = perfectly centered, 0.0 = at corner)
            lesion_centrality = max(0.0, 1.0 - euclidean_distance)
        else:
            lesion_centrality = 0.0

        return float(lesion_size_ratio), float(lesion_centrality)

    except Exception as e:
        # Fallback to edge-based approximation if saliency fails
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150)

        # Estimate lesion as high edge density region
        center_region = edges[h//4:3*h//4, w//4:3*w//4]
        lesion_size_ratio = center_region.sum() / (edges.sum() + 1e-6)
        lesion_centrality = 0.75  # Assume center if we can't detect properly

        return float(lesion_size_ratio), float(lesion_centrality)
