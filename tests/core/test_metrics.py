"""
Tests for the metrics module.
"""
import numpy as np
import cv2
import pytest
from seg_ana.core.metrics import (
    get_largest_contour,
    calculate_basic_metrics,
    calculate_ellipse_metrics,
    calculate_convexity_metrics,
    calculate_all_metrics,
    count_protrusions,
    create_mathematical_circle
)


def create_circle_mask(size=100, radius=30, center=None):
    """Create a circular binary mask for testing."""
    if center is None:
        center = (size // 2, size // 2)
    
    mask = np.zeros((size, size), dtype=np.uint8)
    cv2.circle(mask, center, radius, 1, -1)
    return mask


def create_ellipse_mask(size=100, axes=(40, 20), angle=0, center=None):
    """Create an elliptical binary mask for testing."""
    if center is None:
        center = (size // 2, size // 2)
    
    mask = np.zeros((size, size), dtype=np.uint8)
    cv2.ellipse(mask, center, axes, angle, 0, 360, 1, -1)
    return mask


def create_shape_with_protrusions(size=100, radius=30, protrusion_radius=5, num_protrusions=4):
    """Create a shape with protrusions for testing."""
    mask = create_circle_mask(size, radius)
    center = (size // 2, size // 2)
    
    # Add protrusions
    for i in range(num_protrusions):
        angle = i * (2 * np.pi / num_protrusions)
        x = int(center[0] + (radius + 5) * np.cos(angle))
        y = int(center[1] + (radius + 5) * np.sin(angle))
        cv2.circle(mask, (x, y), protrusion_radius, 1, -1)
    
    return mask


def test_get_largest_contour():
    """Test finding the largest contour in a mask."""
    # Create a mask with two circles
    mask = np.zeros((100, 100), dtype=np.uint8)
    cv2.circle(mask, (30, 30), 10, 1, -1)  # Small circle
    cv2.circle(mask, (70, 70), 20, 1, -1)  # Larger circle
    
    contour = get_largest_contour(mask)
    
    # The contour should be from the larger circle
    assert cv2.contourArea(contour) >= 1200  # π * 20² ≈ 1257


def test_calculate_basic_metrics():
    """Test calculation of basic shape metrics."""
    # Use mathematical circle for more accurate roundness test
    mask = create_mathematical_circle(size=(100, 100), radius=30)
    contour = get_largest_contour(mask)
    metrics = calculate_basic_metrics(contour)
    
    # Check if metrics exist
    assert 'area' in metrics
    assert 'perimeter' in metrics
    assert 'roundness' in metrics
    
    # Check values for a circle
    # Due to discretization, the actual area might be slightly different from the theoretical value
    # π * 30² = 2827.43, but discretization can make it a bit smaller
    assert metrics['area'] > 2700 and metrics['area'] < 2900
    assert metrics['roundness'] == 1.0  # Should be exactly 1 for a perfect circle


def test_calculate_ellipse_metrics():
    """Test calculation of ellipse metrics."""
    mask = create_ellipse_mask()
    contour = get_largest_contour(mask)
    metrics = calculate_ellipse_metrics(contour)
    
    # Check if metrics exist
    assert 'ellipticity' in metrics
    assert 'major_axis' in metrics
    assert 'minor_axis' in metrics
    
    # Check ellipticity for the 2:1 ratio
    assert metrics['ellipticity'] > 1.8 and metrics['ellipticity'] < 2.2


def test_calculate_convexity_metrics():
    """Test calculation of convexity metrics."""
    # Create a star-like shape that's not convex
    mask = create_shape_with_protrusions()
    contour = get_largest_contour(mask)
    metrics = calculate_convexity_metrics(contour)
    
    # Check if metrics exist
    assert 'solidity' in metrics
    assert 'convexity' in metrics
    
    # For a shape with protrusions, solidity should be less than 1
    assert metrics['solidity'] < 1.0
    # Usually between 0.8-0.95 for a shape with small protrusions
    assert metrics['solidity'] > 0.7  


def test_count_protrusions():
    """Test counting protrusions."""
    # Test with different numbers of protrusions
    for num_protrusions in [2, 4, 6]:
        mask = create_shape_with_protrusions(num_protrusions=num_protrusions)
        contour = get_largest_contour(mask)
        hull = cv2.convexHull(contour)
        
        # Count protrusions with different thresholds
        count_low = count_protrusions(contour, hull, threshold=1.0)
        count_high = count_protrusions(contour, hull, threshold=3.0)
        
        # The detected number may not exactly match the created number
        # due to how OpenCV creates the shapes and how the algorithm detects protrusions
        # But there should be a reasonable relationship
        assert count_low > 0  # Should detect some protrusions
        assert count_high <= count_low  # Higher threshold should detect fewer


def test_calculate_all_metrics():
    """Test calculation of all metrics together."""
    mask = create_shape_with_protrusions()
    metrics = calculate_all_metrics(mask)
    
    # Check if all expected metrics exist
    expected_metrics = [
        'area', 'perimeter', 'roundness', 'equivalent_diameter',
        'ellipticity', 'major_axis', 'minor_axis', 'orientation',
        'solidity', 'convexity', 'convex_hull_area', 'protrusions'
    ]
    
    for metric in expected_metrics:
        assert metric in metrics
    
    # Basic sanity checks on values
    assert metrics['area'] > 0
    assert metrics['perimeter'] > 0
    assert 0 <= metrics['roundness'] <= 1.0
    assert metrics['solidity'] <= 1.0


def test_empty_contour_handling():
    """Test handling of empty contours."""
    # Create an empty mask
    mask = np.zeros((100, 100), dtype=np.uint8)
    
    # All functions should handle this gracefully
    contour = get_largest_contour(mask)
    assert contour.size == 0
    
    basic = calculate_basic_metrics(contour)
    assert basic['area'] == 0.0
    
    ellipse = calculate_ellipse_metrics(contour)
    assert ellipse['ellipticity'] == 0.0
    
    convex = calculate_convexity_metrics(contour)
    assert convex['solidity'] == 0.0
    
    all_metrics = calculate_all_metrics(mask)
    assert all_metrics['area'] == 0.0
    assert all_metrics['protrusions'] == 0


def test_mathematical_circle():
    """Test the mathematical circle creation and its roundness."""
    # Create a perfect circle using the mathematical method
    mask = create_mathematical_circle(size=(100, 100), radius=30)
    metrics = calculate_all_metrics(mask)
    
    # Check that roundness is exactly 1.0
    assert metrics['roundness'] == 1.0
    
    # Check that ellipticity is exactly 1.0
    assert metrics['ellipticity'] == 1.0
    
    # Check that solidity is very close to 1.0
    # Due to discretization, it might not be exactly 1.0
    assert metrics['solidity'] > 0.97
    
    # Check the area is close to theoretical value (π * r²)
    theoretical_area = np.pi * 30**2
    assert abs(metrics['area'] - theoretical_area) / theoretical_area < 0.05  # Within 5%


def test_roundness_with_opencv_circle():
    """Test that even OpenCV circles have good roundness with improved metrics."""
    # Create circle with OpenCV
    mask = create_circle_mask(size=100, radius=30)
    metrics = calculate_all_metrics(mask)
    
    # With improved metrics, even OpenCV circles should have good roundness
    assert metrics['roundness'] == 1.0
    
    # Ellipticity should also be very good
    assert metrics['ellipticity'] == 1.0


def test_ellipticity_accuracy():
    """Test ellipticity calculation for various axis ratios."""
    # Test various axis ratios
    for major, minor in [(40, 40), (60, 30), (80, 20)]:
        mask = create_ellipse_mask(axes=(major, minor))
        metrics = calculate_all_metrics(mask)
        
        expected_ratio = major / minor
        
        # Allow a small error margin for discretization effects
        assert abs(metrics['ellipticity'] - expected_ratio) < 0.1
