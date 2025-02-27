"""
Tests for the basic visualization module.
"""
import numpy as np
import pytest
import matplotlib.pyplot as plt
from seg_ana.visualization.basic import (
    visualize_mask,
    visualize_mask_with_metrics,
    create_comparison_figure,
    plot_metric_histogram,
    create_summary_grid
)
from seg_ana.core.metrics import calculate_all_metrics


# Set matplotlib to non-interactive backend for testing
plt.switch_backend('Agg')


def create_test_mask():
    """Create a simple test mask."""
    mask = np.zeros((100, 100), dtype=np.uint8)
    mask[25:75, 25:75] = 1  # Square in the middle
    return mask


def test_visualize_mask():
    """Test basic mask visualization."""
    mask = create_test_mask()
    
    # Test visualization
    fig = visualize_mask(mask)
    assert isinstance(fig, plt.Figure)
    plt.close(fig)
    
    # Test with a title
    fig = visualize_mask(mask, title="Test Mask")
    assert fig.axes[0].get_title() == "Test Mask"
    plt.close(fig)


def test_visualize_mask_with_metrics():
    """Test visualization with metrics."""
    mask = create_test_mask()
    metrics = calculate_all_metrics(mask)
    
    # Test visualization
    fig = visualize_mask_with_metrics(mask, metrics)
    assert isinstance(fig, plt.Figure)
    
    # Check that the figure has two axes (mask and metrics)
    assert len(fig.axes) == 2
    plt.close(fig)


def test_create_comparison_figure():
    """Test comparison figure creation."""
    # Create a few test masks
    masks = [create_test_mask() for _ in range(3)]
    
    # Calculate metrics
    metrics_list = [calculate_all_metrics(m) for m in masks]
    
    # Test comparison figure
    fig = create_comparison_figure(masks, metrics_list, 'roundness')
    assert isinstance(fig, plt.Figure)
    
    # Check that the figure has the right number of axes
    assert len(fig.axes) == len(masks)
    plt.close(fig)


def test_plot_metric_histogram():
    """Test metric histogram creation."""
    # Create a few test masks with different metrics
    metrics_list = [
        {'roundness': 0.5, 'area': 100},
        {'roundness': 0.7, 'area': 150},
        {'roundness': 0.9, 'area': 200}
    ]
    
    # Test histogram
    fig = plot_metric_histogram(metrics_list, 'roundness')
    assert isinstance(fig, plt.Figure)
    plt.close(fig)
    
    # Test with non-existent metric
    fig = plot_metric_histogram(metrics_list, 'nonexistent')
    assert isinstance(fig, plt.Figure)  # Should still return a figure
    plt.close(fig)


def test_create_summary_grid():
    """Test summary grid creation."""
    # Create a few test masks
    masks = np.array([create_test_mask() for _ in range(9)])
    
    # Calculate metrics
    metrics_list = [calculate_all_metrics(m) for m in masks]
    
    # Test grid
    fig = create_summary_grid(masks, metrics_list, num_samples=4)
    assert isinstance(fig, plt.Figure)
    
    # Check that the figure has a grid of axes
    assert len(fig.axes) >= 4
    plt.close(fig)
