"""
Module for basic visualization of segmentation masks and metrics.
"""
import numpy as np
import matplotlib.pyplot as plt
import cv2
import logging
from typing import Dict, List, Union, Optional, Tuple
from pathlib import Path

from ..core.metrics import get_largest_contour

# set up logging
logger = logging.getLogger(__name__)


def visualize_mask(
    mask: np.ndarray,
    contour: Optional[np.ndarray] = None,
    title: str = "Mask Visualization",
    figsize: Tuple[int, int] = (8, 8),
    save_path: Optional[Union[str, Path]] = None
) -> plt.Figure:
    """
    Visualize a single mask with its contour.
    
    Parameters:
    -----------
    mask : np.ndarray
        Binary mask (2D array)
    contour : np.ndarray, optional
        Pre-computed contour, will compute if None
    title : str, default="Mask Visualization"
        Title for the plot
    figsize : tuple, default=(8, 8)
        Figure size in inches
    save_path : str or Path, optional
        Path to save the figure, if provided
        
    Returns:
    --------
    matplotlib.figure.Figure
        Figure object containing the visualization
        
    Example:
    --------
    >>> mask = np.zeros((100, 100), dtype=np.uint8)
    >>> cv2.circle(mask, (50, 50), 20, 1, -1)
    >>> fig = visualize_mask(mask, title="Circle Mask")
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    # Show the mask in grayscale
    ax.imshow(mask, cmap='gray')
    
    # Compute contour if not provided
    if contour is None and np.any(mask):
        contour = get_largest_contour(mask.astype(np.uint8))
    
    # Draw contour if available
    if contour is not None and contour.size > 0:
        # Reshape contour for plotting
        contour_reshaped = contour.squeeze()
        if contour_reshaped.ndim == 2:  # Normal case
            ax.plot(contour_reshaped[:, 0], contour_reshaped[:, 1], 'r-', linewidth=2)
        elif contour_reshaped.ndim == 1 and contour_reshaped.size == 2:  # Single point
            ax.plot(contour_reshaped[0], contour_reshaped[1], 'ro')
    
    ax.set_title(title)
    ax.axis('equal')
    plt.tight_layout()
    
    # Save if path is provided
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    return fig


def visualize_mask_with_metrics(
    mask: np.ndarray,
    metrics: Dict[str, float],
    figsize: Tuple[int, int] = (10, 8),
    save_path: Optional[Union[str, Path]] = None
) -> plt.Figure:
    """
    Visualize a mask with its associated metrics.
    
    Parameters:
    -----------
    mask : np.ndarray
        Binary mask (2D array)
    metrics : dict
        Dictionary of metrics calculated for the mask
    figsize : tuple, default=(10, 8)
        Figure size in inches
    save_path : str or Path, optional
        Path to save the figure, if provided
        
    Returns:
    --------
    matplotlib.figure.Figure
        Figure object containing the visualization
        
    Example:
    --------
    >>> mask = np.zeros((100, 100), dtype=np.uint8)
    >>> cv2.circle(mask, (50, 50), 20, 1, -1)
    >>> metrics = calculate_all_metrics(mask)
    >>> fig = visualize_mask_with_metrics(mask, metrics)
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize, gridspec_kw={'width_ratios': [2, 1]})
    
    # Show the mask in grayscale
    ax1.imshow(mask, cmap='gray')
    ax1.set_title("Mask")
    ax1.axis('equal')
    
    # Compute and draw contour
    if np.any(mask):
        contour = get_largest_contour(mask.astype(np.uint8))
        if contour.size > 0:
            contour_reshaped = contour.squeeze()
            if contour_reshaped.ndim == 2:  # Normal case
                ax1.plot(contour_reshaped[:, 0], contour_reshaped[:, 1], 'r-', linewidth=2)
    
    # Display metrics
    ax2.axis('off')
    metrics_text = "\n".join([f"{k}: {v:.2f}" for k, v in metrics.items()])
    ax2.text(0.05, 0.95, metrics_text, transform=ax2.transAxes,
             verticalalignment='top', fontsize=10)
    ax2.set_title("Metrics")
    
    plt.tight_layout()
    
    # Save if path is provided
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    return fig


def create_comparison_figure(
    masks: List[np.ndarray],
    metrics_list: List[Dict[str, float]],
    metric_name: str,
    num_samples: int = 5,
    figsize: Tuple[int, int] = (15, 10),
    save_path: Optional[Union[str, Path]] = None
) -> plt.Figure:
    """
    Create a comparison figure for a specific metric across multiple masks.
    
    Parameters:
    -----------
    masks : list of np.ndarray
        List of binary masks
    metrics_list : list of dict
        List of metric dictionaries for each mask
    metric_name : str
        Name of the metric to compare
    num_samples : int, default=5
        Number of masks to display (will show extremes and middle)
    figsize : tuple, default=(15, 10)
        Figure size in inches
    save_path : str or Path, optional
        Path to save the figure, if provided
        
    Returns:
    --------
    matplotlib.figure.Figure
        Figure object containing the visualization
        
    Example:
    --------
    >>> masks = load_and_process('path/to/masks.npy')
    >>> metrics_list = analyze_batch(masks)
    >>> fig = create_comparison_figure(masks, metrics_list, 'roundness')
    """
    if not masks or not metrics_list:
        logger.warning("Empty masks or metrics list provided to create_comparison_figure")
        return plt.figure()
    
    # Get metric values
    metric_values = [m.get(metric_name, 0) for m in metrics_list]
    
    # Sort masks by the metric
    sorted_indices = np.argsort(metric_values)
    
    # Select samples (min, max, and evenly spaced in between)
    if len(masks) <= num_samples:
        sample_indices = sorted_indices
    else:
        # Always include min and max
        min_idx = sorted_indices[0]
        max_idx = sorted_indices[-1]
        
        # Get intermediate samples
        middle_indices = []
        if num_samples > 2:
            step = (len(sorted_indices) - 2) / (num_samples - 2)
            for i in range(num_samples - 2):
                idx = int(1 + i * step)
                middle_indices.append(sorted_indices[idx])
        
        sample_indices = [min_idx] + middle_indices + [max_idx]
    
    # Create figure
    fig, axes = plt.subplots(1, len(sample_indices), figsize=figsize)
    if len(sample_indices) == 1:
        axes = [axes]
    
    # Plot each sample
    for i, idx in enumerate(sample_indices):
        axes[i].imshow(masks[idx], cmap='gray')
        value = metric_values[idx]
        axes[i].set_title(f"{metric_name}: {value:.2f}")
        axes[i].axis('off')
    
    plt.tight_layout()
    
    # Save if path is provided
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    return fig


def plot_metric_histogram(
    metrics_list: List[Dict[str, float]],
    metric_name: str,
    bins: int = 20,
    figsize: Tuple[int, int] = (10, 6),
    save_path: Optional[Union[str, Path]] = None
) -> plt.Figure:
    """
    Create a histogram for a specific metric across all masks.
    
    Parameters:
    -----------
    metrics_list : list of dict
        List of metric dictionaries for each mask
    metric_name : str
        Name of the metric to plot
    bins : int, default=20
        Number of histogram bins
    figsize : tuple, default=(10, 6)
        Figure size in inches
    save_path : str or Path, optional
        Path to save the figure, if provided
        
    Returns:
    --------
    matplotlib.figure.Figure
        Figure object containing the histogram
        
    Example:
    --------
    >>> metrics_list = analyze_batch(masks)
    >>> fig = plot_metric_histogram(metrics_list, 'roundness')
    """
    if not metrics_list:
        logger.warning("Empty metrics list provided to plot_metric_histogram")
        return plt.figure()
    
    # Extract metric values
    values = [m.get(metric_name, np.nan) for m in metrics_list]
    values = [v for v in values if not np.isnan(v)]
    
    if not values:
        logger.warning(f"No valid values found for metric '{metric_name}'")
        return plt.figure()
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Plot histogram
    ax.hist(values, bins=bins, alpha=0.7, color='skyblue', edgecolor='black')
    
    # Add statistics
    mean_val = np.mean(values)
    median_val = np.median(values)
    std_val = np.std(values)
    
    ax.axvline(mean_val, color='red', linestyle='dashed', linewidth=1, label=f'Mean: {mean_val:.2f}')
    ax.axvline(median_val, color='green', linestyle='dashed', linewidth=1, label=f'Median: {median_val:.2f}')
    
    # Add labels and legend
    ax.set_xlabel(metric_name.capitalize())
    ax.set_ylabel('Frequency')
    ax.set_title(f'Distribution of {metric_name.capitalize()}')
    ax.legend()
    
    # Add text with statistics
    stats_text = f"Mean: {mean_val:.2f}\nMedian: {median_val:.2f}\nStd: {std_val:.2f}"
    ax.text(0.95, 0.95, stats_text, transform=ax.transAxes,
            verticalalignment='top', horizontalalignment='right',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.5))
    
    plt.tight_layout()
    
    # Save if path is provided
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    return fig


def create_summary_grid(
    masks: np.ndarray,
    metrics_list: List[Dict[str, float]],
    num_samples: int = 16,
    figsize: Tuple[int, int] = (12, 12),
    sort_by: str = 'area',
    save_path: Optional[Union[str, Path]] = None
) -> plt.Figure:
    """
    Create a grid of mask thumbnails, sorted by a specified metric.
    
    Parameters:
    -----------
    masks : np.ndarray
        Array of binary masks with shape (n_masks, height, width)
    metrics_list : list of dict
        List of metric dictionaries for each mask
    num_samples : int, default=16
        Number of masks to display in the grid
    figsize : tuple, default=(12, 12)
        Figure size in inches
    sort_by : str, default='area'
        Metric to sort the masks by
    save_path : str or Path, optional
        Path to save the figure, if provided
        
    Returns:
    --------
    matplotlib.figure.Figure
        Figure object containing the grid visualization
        
    Example:
    --------
    >>> masks = load_and_process('path/to/masks.npy')
    >>> metrics_list = analyze_batch(masks)
    >>> fig = create_summary_grid(masks, metrics_list, sort_by='roundness')
    """
    if not len(masks) or not metrics_list:
        logger.warning("Empty masks or metrics list provided to create_summary_grid")
        return plt.figure()
    
    # Get metric values and sort
    metric_values = [m.get(sort_by, 0) for m in metrics_list]
    sorted_indices = np.argsort(metric_values)[::-1]  # Descending order
    
    # Determine grid dimensions
    grid_size = int(np.ceil(np.sqrt(num_samples)))
    
    # Select samples
    if len(masks) <= num_samples:
        sample_indices = sorted_indices[:len(masks)]
    else:
        # Use evenly spaced samples
        step = len(sorted_indices) / num_samples
        sample_indices = [sorted_indices[int(i * step)] for i in range(num_samples)]
    
    # Create figure
    fig, axes = plt.subplots(grid_size, grid_size, figsize=figsize)
    axes = axes.flatten()
    
    # Plot each sample
    for i, idx in enumerate(sample_indices):
        if i < len(axes):
            axes[i].imshow(masks[idx], cmap='gray')
            value = metric_values[idx]
            axes[i].set_title(f"{sort_by}: {value:.2f}", fontsize=8)
            axes[i].axis('off')
    
    # Turn off any unused axes
    for i in range(len(sample_indices), len(axes)):
        axes[i].axis('off')
    
    plt.tight_layout()
    
    # Save if path is provided
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    return fig
