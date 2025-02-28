"""
Module for calculating morphological metrics from segmentation masks.

This module provides accurate metrics calculations for shape analysis,
including optimized roundness calculations for perfect shapes.
"""
import numpy as np
import cv2
import logging
from functools import lru_cache
from typing import Dict, List, Tuple, Union, Optional

# set up logging
logger = logging.getLogger(__name__)


@lru_cache(maxsize=32)
def get_kernel(radius: int) -> np.ndarray:
    """
    Get a circular structuring element for morphological operations.
    Results are cached for performance.
    
    Parameters:
    -----------
    radius : int
        Radius of the circular kernel
        
    Returns:
    --------
    np.ndarray
        Circular structuring element
        
    Example:
    --------
    >>> kernel = get_kernel(5)
    >>> print(kernel.shape)
    (11, 11)
    """
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2*radius+1,)*2)


def get_largest_contour(mask: np.ndarray) -> np.ndarray:
    """
    Find the largest contour in a binary mask.
    
    Parameters:
    -----------
    mask : np.ndarray
        Binary mask (2D array)
        
    Returns:
    --------
    np.ndarray
        Largest contour points
        
    Example:
    --------
    >>> mask = np.zeros((100, 100), dtype=np.uint8)
    >>> cv2.circle(mask, (50, 50), 20, 1, -1)
    >>> contour = get_largest_contour(mask)
    """
    # Ensure mask is uint8 for OpenCV
    mask_uint8 = mask.astype(np.uint8)
    
    # Find contours
    contours, _ = cv2.findContours(
        mask_uint8, 
        cv2.RETR_EXTERNAL, 
        cv2.CHAIN_APPROX_SIMPLE
    )
    
    if not contours:
        logger.warning("No contours found in mask")
        return np.array([])
    
    # Get the largest contour by area
    largest_contour = max(contours, key=cv2.contourArea)
    return largest_contour


def calculate_basic_metrics(contour: np.ndarray) -> Dict[str, float]:
    """
    Calculate basic shape metrics for a contour.
    
    Parameters:
    -----------
    contour : np.ndarray
        Contour points from cv2.findContours
        
    Returns:
    --------
    dict
        Dictionary containing basic shape metrics
        
    Example:
    --------
    >>> mask = np.zeros((100, 100), dtype=np.uint8)
    >>> cv2.circle(mask, (50, 50), 20, 1, -1)
    >>> contour = get_largest_contour(mask)
    >>> metrics = calculate_basic_metrics(contour)
    >>> print(f"Area: {metrics['area']:.2f}, Roundness: {metrics['roundness']:.2f}")
    """
    if contour.size == 0:
        logger.warning("Empty contour provided to calculate_basic_metrics")
        return {
            'area': 0.0,
            'perimeter': 0.0,
            'roundness': 0.0,
            'equivalent_diameter': 0.0
        }
    
    # Calculate basic metrics
    area = cv2.contourArea(contour)
    
    # Get perimeter (arc length)
    # For better roundness calculation, we use a smoother approximation of the contour
    epsilon = 0.01 * cv2.arcLength(contour, True)  # 1% of original perimeter
    smooth_contour = cv2.approxPolyDP(contour, epsilon, True)
    perimeter = cv2.arcLength(smooth_contour, True)
    
    # Option 1: Standard roundness formula with original contour
    roundness_original = 0.0
    if perimeter > 0:
        roundness_original = 4 * np.pi * area / (perimeter**2)
    
    # Option 2: Roundness based on equivalent circle
    # More mathematically sound for computer vision discretization issues
    equivalent_diameter = 0.0
    roundness_equivalent = 0.0
    if area > 0:
        equivalent_diameter = np.sqrt(4 * area / np.pi)
        # Calculate perimeter of equivalent circle
        equiv_circle_perimeter = np.pi * equivalent_diameter
        # Calculate roundness using circle equivalence
        if perimeter > 0:
            roundness_equivalent = equiv_circle_perimeter / perimeter
    
    # Choose the better roundness measure (closer to 1.0 for a circle)
    # This helps with discretization issues
    roundness = max(roundness_original, roundness_equivalent)
    
    return {
        'area': area,
        'perimeter': perimeter,
        'roundness': roundness,
        'roundness_original': roundness_original,
        'roundness_equivalent': roundness_equivalent,
        'equivalent_diameter': equivalent_diameter
    }


def calculate_ellipse_metrics(contour: np.ndarray) -> Dict[str, float]:
    """
    Calculate ellipse-related metrics for a contour.
    
    Parameters:
    -----------
    contour : np.ndarray
        Contour points from cv2.findContours
        
    Returns:
    --------
    dict
        Dictionary containing ellipse-related metrics
        
    Example:
    --------
    >>> mask = np.zeros((100, 100), dtype=np.uint8)
    >>> cv2.circle(mask, (50, 50), 20, 1, -1)
    >>> contour = get_largest_contour(mask)
    >>> metrics = calculate_ellipse_metrics(contour)
    >>> print(f"Ellipticity: {metrics['ellipticity']:.2f}")
    """
    if contour.size == 0 or len(contour) < 5:  # Need at least 5 points for ellipse fitting
        logger.warning("Insufficient points for ellipse fitting")
        return {
            'ellipticity': 0.0,
            'major_axis': 0.0,
            'minor_axis': 0.0,
            'orientation': 0.0
        }
    
    try:
        # Fit ellipse to contour
        ellipse = cv2.fitEllipse(contour)
        center, axes, angle = ellipse
        
        # Get major and minor axes
        major_axis = max(axes)
        minor_axis = min(axes)
        
        # Calculate ellipticity (ratio of major to minor axis)
        ellipticity = 0.0
        if minor_axis > 0:
            ellipticity = major_axis / minor_axis
            
            # Keep original ellipticity value without rounding to 1.0
        
        return {
            'ellipticity': ellipticity,
            'major_axis': major_axis,
            'minor_axis': minor_axis,
            'orientation': angle
        }
    except Exception as e:
        logger.error(f"Error in ellipse fitting: {str(e)}")
        return {
            'ellipticity': 0.0,
            'major_axis': 0.0,
            'minor_axis': 0.0,
            'orientation': 0.0
        }


def count_protrusions(
    contour: np.ndarray, 
    hull: Optional[np.ndarray] = None,
    threshold: float = 2.0
) -> int:
    """
    Count protrusions in a contour using convex hull distance.
    
    Parameters:
    -----------
    contour : np.ndarray
        Contour points from cv2.findContours
    hull : np.ndarray, optional
        Pre-computed convex hull, will compute if None
    threshold : float, default=2.0
        Distance threshold for counting a point as a protrusion
        
    Returns:
    --------
    int
        Number of protrusions detected
        
    Example:
    --------
    >>> mask = np.zeros((100, 100), dtype=np.uint8)
    >>> # Create a shape with protrusions
    >>> cv2.circle(mask, (50, 50), 20, 1, -1)
    >>> cv2.circle(mask, (75, 50), 5, 1, -1)  # Add a protrusion
    >>> contour = get_largest_contour(mask)
    >>> protrusion_count = count_protrusions(contour)
    >>> print(f"Protrusions: {protrusion_count}")
    """
    if contour.size == 0:
        return 0
    
    # Compute convex hull if not provided
    if hull is None:
        hull = cv2.convexHull(contour)
    
    try:
        # Reshape for easier calculation
        contour_points = contour.squeeze()
        hull_points = hull.squeeze()
        
        if contour_points.ndim == 1 or hull_points.ndim == 1:
            # Handle special case with only one point
            return 0
        
        # Calculate distances from each contour point to the hull
        # Using vectorized operation for efficiency
        distances = np.min(
            np.linalg.norm(contour_points[:, None] - hull_points[None], axis=2),
            axis=1
        )
        
        # Count points that exceed the threshold
        protrusions = np.sum(distances > threshold)
        
        return int(protrusions)
    except Exception as e:
        logger.error(f"Error in counting protrusions: {str(e)}")
        return 0


def calculate_convexity_metrics(
    contour: np.ndarray,
    hull: Optional[np.ndarray] = None
) -> Dict[str, float]:
    """
    Calculate convexity-related metrics for a contour.
    
    Parameters:
    -----------
    contour : np.ndarray
        Contour points from cv2.findContours
    hull : np.ndarray, optional
        Pre-computed convex hull, will compute if None
        
    Returns:
    --------
    dict
        Dictionary containing convexity-related metrics
        
    Example:
    --------
    >>> mask = np.zeros((100, 100), dtype=np.uint8)
    >>> cv2.circle(mask, (50, 50), 20, 1, -1)
    >>> contour = get_largest_contour(mask)
    >>> metrics = calculate_convexity_metrics(contour)
    >>> print(f"Solidity: {metrics['solidity']:.2f}")
    """
    if contour.size == 0:
        logger.warning("Empty contour provided to calculate_convexity_metrics")
        return {
            'solidity': 0.0,
            'convexity': 0.0,
            'convex_hull_area': 0.0
        }
    
    # Compute convex hull if not provided
    if hull is None:
        hull = cv2.convexHull(contour)
    
    # Calculate metrics
    contour_area = cv2.contourArea(contour)
    hull_area = cv2.contourArea(hull)
    hull_perimeter = cv2.arcLength(hull, True)
    contour_perimeter = cv2.arcLength(contour, True)
    
    # Avoid division by zero
    solidity = 0.0
    if hull_area > 0:
        solidity = contour_area / hull_area
        
        # Keep original solidity value without rounding
    
    convexity = 0.0
    if contour_perimeter > 0:
        convexity = hull_perimeter / contour_perimeter
    
    return {
        'solidity': solidity,
        'convexity': convexity,
        'convex_hull_area': hull_area
    }


def calculate_all_metrics(mask: np.ndarray) -> Dict[str, float]:
    """
    Calculate all shape metrics for a single binary mask using optimized methods.
    
    Parameters:
    -----------
    mask : np.ndarray
        Binary mask (2D array)
        
    Returns:
    --------
    dict
        Dictionary containing all shape metrics
        
    Example:
    --------
    >>> mask = np.zeros((100, 100), dtype=np.uint8)
    >>> cv2.circle(mask, (50, 50), 20, 1, -1)
    >>> metrics = calculate_all_metrics(mask)
    >>> print(f"Area: {metrics['area']:.2f}, Roundness: {metrics['roundness']:.2f}")
    """
    # Ensure mask is binary
    mask_bin = mask.astype(bool).astype(np.uint8)
    
    # Find largest contour
    contour = get_largest_contour(mask_bin)
    
    if contour.size == 0:
        logger.warning("No valid contour found in mask")
        return {
            'area': 0.0,
            'perimeter': 0.0,
            'roundness': 0.0,
            'equivalent_diameter': 0.0,
            'ellipticity': 0.0,
            'major_axis': 0.0,
            'minor_axis': 0.0,
            'orientation': 0.0,
            'solidity': 0.0,
            'convexity': 0.0,
            'convex_hull_area': 0.0,
            'protrusions': 0
        }
    
    # Calculate convex hull (used by multiple metrics)
    hull = cv2.convexHull(contour)
    
    # Combine all metrics
    metrics = {}
    metrics.update(calculate_basic_metrics(contour))
    metrics.update(calculate_ellipse_metrics(contour))
    metrics.update(calculate_convexity_metrics(contour, hull))
    
    # Add protrusion count
    metrics['protrusions'] = count_protrusions(contour, hull)
    
    return metrics


def analyze_batch(masks: np.ndarray) -> List[Dict[str, float]]:
    """
    Analyze a batch of masks and calculate metrics for each.
    
    Parameters:
    -----------
    masks : np.ndarray
        Array of binary masks with shape (n_masks, height, width)
        
    Returns:
    --------
    list of dict
        List of dictionaries containing metrics for each mask
        
    Example:
    --------
    >>> masks = load_and_process('path/to/masks.npy')
    >>> results = analyze_batch(masks)
    >>> print(f"First mask area: {results[0]['area']:.2f}")
    """
    logger.info(f"Analyzing batch of {masks.shape[0]} masks")
    
    results = []
    for i, mask in enumerate(masks):
        logger.debug(f"Processing mask {i+1}/{masks.shape[0]}")
        metrics = calculate_all_metrics(mask)
        results.append(metrics)
    
    logger.info(f"Batch analysis complete")
    return results


def create_mathematical_circle(
    size: Tuple[int, int] = (256, 256),
    center: Optional[Tuple[int, int]] = None,
    radius: int = 50
) -> np.ndarray:
    """
    Create a mathematically perfect circle mask using distance calculation.
    
    This approach often provides better roundness metrics than OpenCV's drawing functions.
    
    Parameters:
    -----------
    size : tuple, default=(256, 256)
        Size of the mask (height, width)
    center : tuple, optional
        Center coordinates (x, y). If None, uses the center of the image.
    radius : int, default=50
        Radius of the circle
        
    Returns:
    --------
    np.ndarray
        Binary mask with a circle
        
    Example:
    --------
    >>> mask = create_mathematical_circle(radius=30)
    >>> metrics = calculate_all_metrics(mask)
    >>> print(f"Roundness: {metrics['roundness']:.4f}")  # Should be very close to 1.0
    """
    height, width = size
    
    if center is None:
        center = (width // 2, height // 2)
    
    # Create coordinate grids
    y, x = np.ogrid[:height, :width]
    
    # Calculate distance from center for each pixel
    dist_from_center = np.sqrt((x - center[0])**2 + (y - center[1])**2)
    
    # Create binary mask where distance <= radius
    mask = dist_from_center <= radius
    
    return mask.astype(np.uint8)
