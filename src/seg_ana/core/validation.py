"""
Module for validating and exporting mask data for visual inspection.
"""
import numpy as np
import matplotlib.pyplot as plt
import cv2
from pathlib import Path
import logging
from typing import Optional, Union, Tuple, Dict

from .metrics import calculate_all_metrics, get_largest_contour

# Setup logging
logger = logging.getLogger(__name__)

def export_mask_visualization(
    mask: np.ndarray, 
    filename: Union[str, Path],
    show_contour: bool = True,
    show_metrics: bool = True,
    figsize: Tuple[int, int] = (10, 8),
) -> str:
    """
    Export a visualization of a mask with optional contour and metrics.
    
    Parameters:
    -----------
    mask : np.ndarray
        Binary mask to visualize
    filename : str or Path
        Filename to save the visualization to
    show_contour : bool, default=True
        Whether to show the contour on the mask
    show_metrics : bool, default=True
        Whether to show calculated metrics
    figsize : tuple, default=(10, 8)
        Figure size in inches
        
    Returns:
    --------
    str
        Path to the saved visualization
        
    Example:
    --------
    >>> from seg_ana.core.synthetic import create_circle_mask
    >>> from seg_ana.core.validation import export_mask_visualization
    >>> mask = create_circle_mask(size=(256, 256), radius=50)
    >>> export_mask_visualization(mask, "circle_visualization.png")
    """
    # Convert to Path object for easier handling
    filename = Path(filename)
    
    # Create figure
    if show_metrics:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize, 
                                      gridspec_kw={'width_ratios': [2, 1]})
    else:
        fig, ax1 = plt.subplots(figsize=figsize)
    
    # Show mask
    ax1.imshow(mask, cmap='gray')
    ax1.set_title("Mask")
    
    # Draw contour if requested
    if show_contour and np.any(mask):
        contour = get_largest_contour(mask.astype(np.uint8))
        if contour.size > 0:
            contour_reshaped = contour.squeeze()
            if contour_reshaped.ndim == 2:  # Normal case
                ax1.plot(contour_reshaped[:, 0], contour_reshaped[:, 1], 'r-', linewidth=2)
            elif contour_reshaped.ndim == 1 and contour_reshaped.size == 2:  # Single point
                ax1.plot(contour_reshaped[0], contour_reshaped[1], 'ro')
    
    # Show metrics if requested
    if show_metrics:
        metrics = calculate_all_metrics(mask)
        ax2.axis('off')
        metrics_text = "\n".join([f"{k}: {v:.4f}" for k, v in metrics.items()])
        ax2.text(0.05, 0.95, metrics_text, transform=ax2.transAxes,
                verticalalignment='top', fontsize=10)
        ax2.set_title("Metrics")
    
    plt.tight_layout()
    
    # Create directory if it doesn't exist
    filename.parent.mkdir(parents=True, exist_ok=True)
    
    # Save figure
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    plt.close()
    
    return str(filename)

def export_mask_with_analyzed_contour(
    mask: np.ndarray,
    filename: Union[str, Path],
    figsize: Tuple[int, int] = (12, 10)
) -> str:
    """
    Export a detailed visualization of a mask with contour analysis.
    
    Parameters:
    -----------
    mask : np.ndarray
        Binary mask to visualize
    filename : str or Path
        Filename to save the visualization to
    figsize : tuple, default=(12, 10)
        Figure size in inches
        
    Returns:
    --------
    str
        Path to the saved visualization
        
    Example:
    --------
    >>> from seg_ana.core.synthetic import create_circle_mask
    >>> from seg_ana.core.validation import export_mask_with_analyzed_contour
    >>> mask = create_circle_mask(size=(256, 256), radius=50)
    >>> export_mask_with_analyzed_contour(mask, "circle_analysis.png")
    """
    # Convert to Path object for easier handling
    filename = Path(filename)
    
    # Get metrics and contour
    metrics = calculate_all_metrics(mask)
    contour = get_largest_contour(mask.astype(np.uint8))
    
    if contour.size == 0:
        logger.warning("No contour found for mask analysis")
        return ""
    
    # Reshape contour for analysis
    contour_points = contour.squeeze()
    
    # Calculate centroid of shape
    M = cv2.moments(contour)
    if M["m00"] != 0:
        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])
    else:
        # If moments method fails, use simple average
        cx = np.mean(contour_points[:, 0])
        cy = np.mean(contour_points[:, 1])
    
    # Calculate distances from centroid to each contour point
    distances = np.sqrt((contour_points[:, 0] - cx)**2 + (contour_points[:, 1] - cy)**2)
    
    # Create figure with 2x2 grid
    fig, axes = plt.subplots(2, 2, figsize=figsize, gridspec_kw={'height_ratios': [2, 1]})
    
    # Top left: Mask with contour
    axes[0, 0].imshow(mask, cmap='gray')
    axes[0, 0].plot(contour_points[:, 0], contour_points[:, 1], 'r-', linewidth=2)
    axes[0, 0].plot(cx, cy, 'go', markersize=8)  # Mark centroid
    axes[0, 0].set_title("Mask with Contour")
    
    # Top right: Distance map visualization
    axes[0, 1].scatter(np.arange(len(distances)), distances, alpha=0.5)
    axes[0, 1].axhline(y=np.mean(distances), color='r', linestyle='--', label=f'Mean: {np.mean(distances):.2f}')
    axes[0, 1].set_title("Distance from Centroid")
    axes[0, 1].set_xlabel("Contour Point Index")
    axes[0, 1].set_ylabel("Distance (pixels)")
    axes[0, 1].legend()
    
    # Bottom left: Metrics
    axes[1, 0].axis('off')
    metrics_text = "\n".join([f"{k}: {v:.4f}" for k, v in metrics.items()])
    axes[1, 0].text(0.05, 0.95, metrics_text, transform=axes[1, 0].transAxes,
                   verticalalignment='top', fontsize=10)
    axes[1, 0].set_title("Metrics")
    
    # Bottom right: Contour statistics
    axes[1, 1].axis('off')
    stats_text = [
        f"Number of contour points: {len(contour_points)}",
        f"Mean distance: {np.mean(distances):.2f}",
        f"Min distance: {np.min(distances):.2f}",
        f"Max distance: {np.max(distances):.2f}",
        f"Std dev of distance: {np.std(distances):.2f}",
        f"CV of distance (%): {np.std(distances)/np.mean(distances)*100:.2f}%",
        f"Theoretical roundness: 1.0",
        f"Calculated roundness: {metrics['roundness']:.4f}",
        f"Difference: {abs(1.0 - metrics['roundness'])*100:.2f}%"
    ]
    stats_text = "\n".join(stats_text)
    axes[1, 1].text(0.05, 0.95, stats_text, transform=axes[1, 1].transAxes,
                   verticalalignment='top', fontsize=10)
    axes[1, 1].set_title("Contour Statistics")
    
    plt.tight_layout()
    
    # Create directory if it doesn't exist
    filename.parent.mkdir(parents=True, exist_ok=True)
    
    # Save figure
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    plt.close()
    
    return str(filename)

def save_mask_to_file(
    mask: np.ndarray,
    filename: Union[str, Path],
    include_png: bool = True
) -> Tuple[str, Optional[str]]:
    """
    Save a binary mask to a NPY file and optionally a PNG image.
    
    Parameters:
    -----------
    mask : np.ndarray
        Binary mask to save
    filename : str or Path
        Base filename (without extension) to save to
    include_png : bool, default=True
        Whether to also save a PNG version of the mask
        
    Returns:
    --------
    tuple
        Paths to the saved NPY file and PNG file (if requested)
        
    Example:
    --------
    >>> from seg_ana.core.synthetic import create_circle_mask
    >>> from seg_ana.core.validation import save_mask_to_file
    >>> mask = create_circle_mask(size=(256, 256), radius=50)
    >>> npy_path, png_path = save_mask_to_file(mask, "circle_mask")
    """
    # Convert to Path object for easier handling
    filename = Path(filename)
    
    # Create directory if it doesn't exist
    filename.parent.mkdir(parents=True, exist_ok=True)
    
    # Save NPY file
    npy_path = filename.with_suffix('.npy')
    np.save(npy_path, mask)
    
    # Save PNG if requested
    png_path = None
    if include_png:
        png_path = filename.with_suffix('.png')
        plt.figure(figsize=(8, 8))
        plt.imshow(mask, cmap='gray')
        plt.axis('off')
        plt.tight_layout()
        plt.savefig(png_path, dpi=150, bbox_inches='tight')
        plt.close()
    
    return str(npy_path), str(png_path) if png_path else None

def create_mathematical_circle(
    size: Tuple[int, int] = (256, 256),
    center: Optional[Tuple[int, int]] = None,
    radius: int = 50
) -> np.ndarray:
    """
    Create a perfect circle mask using mathematical calculation.
    
    This approach may provide a more perfect circle than OpenCV's drawing functions.
    
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
        Binary mask with a mathematically perfect circle
        
    Example:
    --------
    >>> from seg_ana.core.validation import create_mathematical_circle
    >>> mask = create_mathematical_circle(radius=30)
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

def compare_circle_methods(
    radius: int = 50,
    size: Tuple[int, int] = (256, 256),
    save_path: Optional[Union[str, Path]] = None
) -> Dict[str, float]:
    """
    Compare different methods of creating a circle and their roundness.
    
    Parameters:
    -----------
    radius : int, default=50
        Radius of the circle
    size : tuple, default=(256, 256)
        Size of the mask (height, width)
    save_path : str or Path, optional
        Directory to save comparison images, if provided
        
    Returns:
    --------
    dict
        Dictionary with roundness values for each method
        
    Example:
    --------
    >>> from seg_ana.core.validation import compare_circle_methods
    >>> results = compare_circle_methods(save_path="./circle_comparison")
    >>> print(f"OpenCV roundness: {results['opencv']}")
    >>> print(f"Mathematical roundness: {results['mathematical']}")
    """
    from .synthetic import create_circle_mask
    
    # Method 1: OpenCV circle drawing (from synthetic.py)
    mask_cv = create_circle_mask(size=size, radius=radius, noise=0.0)
    
    # Method 2: Mathematical distance calculation
    mask_math = create_mathematical_circle(size=size, radius=radius)
    
    # Calculate metrics for both
    metrics_cv = calculate_all_metrics(mask_cv)
    metrics_math = calculate_all_metrics(mask_math)
    
    roundness_cv = metrics_cv['roundness']
    roundness_math = metrics_math['roundness']
    
    # Save comparison if path provided
    if save_path:
        save_path = Path(save_path)
        save_path.mkdir(parents=True, exist_ok=True)
        
        # Save individual masks
        save_mask_to_file(mask_cv, save_path / "circle_opencv")
        save_mask_to_file(mask_math, save_path / "circle_mathematical")
        
        # Save visualizations
        export_mask_visualization(
            mask_cv, 
            save_path / "circle_opencv_viz.png"
        )
        export_mask_visualization(
            mask_math, 
            save_path / "circle_mathematical_viz.png"
        )
        
        # Create comparison figure
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
        
        ax1.imshow(mask_cv, cmap='gray')
        ax1.set_title(f"OpenCV Circle\nRoundness: {roundness_cv:.4f}")
        
        ax2.imshow(mask_math, cmap='gray')
        ax2.set_title(f"Mathematical Circle\nRoundness: {roundness_math:.4f}")
        
        plt.tight_layout()
        plt.savefig(save_path / 'circle_methods_comparison.png', dpi=150)
        plt.close()
    
    return {
        'opencv': roundness_cv,
        'mathematical': roundness_math
    }
