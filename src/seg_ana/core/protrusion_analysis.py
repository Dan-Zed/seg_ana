"""
Enhanced protrusion analysis module.

This module provides functions to isolate and analyze individual protrusions
from a shape, treating them as "arms" extending from a main body.
"""

import numpy as np
import cv2
import matplotlib.pyplot as plt
from pathlib import Path
import logging
from typing import List, Dict, Tuple, Optional, Union

# Set up logging
logger = logging.getLogger(__name__)


def isolate_protrusions(
    mask: np.ndarray,
    min_area_ratio: float = 0.01,
    max_area_ratio: float = 0.2
) -> Tuple[np.ndarray, List[np.ndarray], np.ndarray]:
    """
    Isolate the main body and individual protrusions from a mask.
    
    This approach uses morphological operations to identify a core "body"
    and then isolates extending "arms" as separate protrusions.
    
    Parameters:
    -----------
    mask : np.ndarray
        Binary mask of the shape
    min_area_ratio : float, default=0.01
        Minimum area ratio (to original shape) for a region to be considered a protrusion
    max_area_ratio : float, default=0.2
        Maximum area ratio for a region to be considered a protrusion
        
    Returns:
    --------
    tuple
        (body_mask, list of protrusion_masks, combined visualization)
    """
    # Ensure binary mask
    mask_bin = mask.astype(bool).astype(np.uint8)
    
    # Get original contour area for reference
    contours, _ = cv2.findContours(
        mask_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    if not contours:
        logger.warning("No contours found in mask")
        return mask_bin, [], mask_bin
    
    original_contour = max(contours, key=cv2.contourArea)
    original_area = cv2.contourArea(original_contour)
    
    # Create a structuring element for erosion
    # Use a fairly large element to isolate the main body
    radius = int(np.sqrt(original_area / np.pi) * 0.4)  # 40% of equivalent circle radius
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2*radius+1, 2*radius+1))
    
    # Erode the mask to get the main body
    body = cv2.erode(mask_bin, kernel)
    
    # Dilate the body slightly to create a better separation
    body_dilated = cv2.dilate(body, kernel)
    
    # Isolate protrusions: original mask - dilated body
    protrusions_mask = cv2.subtract(mask_bin, body_dilated)
    
    # Find connected components in the protrusions mask
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
        protrusions_mask, connectivity=8
    )
    
    # Filter components based on area
    min_area = original_area * min_area_ratio
    max_area = original_area * max_area_ratio
    
    # List to hold individual protrusion masks
    protrusion_masks = []
    
    # Create a visualization image
    # RGB image: blue=body, green=protrusions, red=unused small components
    visualization = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)
    visualization[:, :, 0] = body_dilated * 255  # Body in blue
    
    # Process each connected component
    for i in range(1, num_labels):  # Skip background (0)
        area = stats[i, cv2.CC_STAT_AREA]
        
        # Check if this component is a valid protrusion
        if min_area <= area <= max_area:
            # Create a mask for this protrusion
            protrusion_mask = np.zeros_like(mask_bin)
            protrusion_mask[labels == i] = 1
            protrusion_masks.append(protrusion_mask)
            
            # Add to visualization (green channel)
            visualization[:, :, 1] += protrusion_mask * 255
        else:
            # Small components in red channel
            small_component = np.zeros_like(mask_bin)
            small_component[labels == i] = 1
            visualization[:, :, 2] += small_component * 255
    
    # Ensure we don't exceed 255 in any channel
    visualization = np.clip(visualization, 0, 255)
    
    return body_dilated, protrusion_masks, visualization


def analyze_protrusion(
    protrusion_mask: np.ndarray,
    body_mask: np.ndarray
) -> Dict[str, float]:
    """
    Analyze a single protrusion to extract metrics.
    
    Parameters:
    -----------
    protrusion_mask : np.ndarray
        Binary mask of the protrusion
    body_mask : np.ndarray
        Binary mask of the main body
        
    Returns:
    --------
    dict
        Dictionary of protrusion metrics
    """
    # Get contour of the protrusion
    contours, _ = cv2.findContours(
        protrusion_mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    if not contours:
        logger.warning("No contour found for protrusion")
        return {
            'area': 0,
            'length': 0,
            'width': 0,
            'aspect_ratio': 0,
            'distance_from_center': 0,
            'angle': 0
        }
    
    protrusion_contour = max(contours, key=cv2.contourArea)
    
    # Get contour of the body
    body_contours, _ = cv2.findContours(
        body_mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    body_contour = max(body_contours, key=cv2.contourArea)
    
    # Calculate protrusion area
    area = cv2.contourArea(protrusion_contour)
    
    # Find the connection point between protrusion and body
    # This is approximated as the closest point on the protrusion to the body
    min_distance = float('inf')
    connection_point = None
    
    # Protrusion points
    protrusion_points = protrusion_contour.squeeze()
    
    # Make sure we handle the case of a single point
    if protrusion_points.ndim == 1:
        protrusion_points = protrusion_points.reshape(1, 2)
    
    # Body contour points
    body_points = body_contour.squeeze()
    
    # Find the connection point (closest point to body)
    for point in protrusion_points:
        # Calculate distance to body
        point_distances = np.sqrt(np.sum((body_points - point)**2, axis=1))
        current_min = np.min(point_distances)
        if current_min < min_distance:
            min_distance = current_min
            connection_point = point
    
    if connection_point is None:
        logger.warning("Could not find connection point for protrusion")
        connection_point = protrusion_points[0]
    
    # Get the farthest point from the connection point
    max_distance = 0
    farthest_point = None
    
    for point in protrusion_points:
        dist = np.sqrt(np.sum((point - connection_point)**2))
        if dist > max_distance:
            max_distance = dist
            farthest_point = point
    
    if farthest_point is None:
        logger.warning("Could not find farthest point for protrusion")
        farthest_point = protrusion_points[-1]
    
    # Calculate protrusion length (connection to farthest point)
    length = max_distance
    
    # Find the body center
    M = cv2.moments(body_contour)
    if M["m00"] != 0:
        body_center_x = int(M["m10"] / M["m00"])
        body_center_y = int(M["m01"] / M["m00"])
        body_center = np.array([body_center_x, body_center_y])
    else:
        # Fallback to average of points
        body_center = np.mean(body_points, axis=0)
    
    # Calculate distance from body center to protrusion tip
    distance_from_center = np.sqrt(np.sum((farthest_point - body_center)**2))
    
    # Calculate angle of the protrusion (from body center to tip)
    dx = farthest_point[0] - body_center[0]
    dy = farthest_point[1] - body_center[1]
    angle = np.degrees(np.arctan2(dy, dx))
    
    # Normalize angle to 0-360
    if angle < 0:
        angle += 360
    
    # Estimate protrusion width
    # We'll use the minimum bounding rectangle approach
    rect = cv2.minAreaRect(protrusion_contour)
    _, (width, height), _ = rect
    
    # Use the smaller dimension as width
    width = min(width, height)
    
    # Calculate aspect ratio of the protrusion
    aspect_ratio = length / width if width > 0 else 0
    
    return {
        'area': area,
        'length': length,
        'width': width,
        'aspect_ratio': aspect_ratio,
        'distance_from_center': distance_from_center,
        'angle': angle
    }


def analyze_all_protrusions(
    mask: np.ndarray,
    visualize: bool = False,
    output_dir: Optional[Union[str, Path]] = None
) -> Dict[str, Union[int, List[Dict[str, float]]]]:
    """
    Isolate and analyze all protrusions in a mask.
    
    Parameters:
    -----------
    mask : np.ndarray
        Binary mask of the shape
    visualize : bool, default=False
        Whether to generate visualization images
    output_dir : str or Path, optional
        Directory to save visualizations (required if visualize=True)
        
    Returns:
    --------
    dict
        Dictionary with protrusion count and individual protrusion metrics
    """
    if visualize and output_dir is None:
        raise ValueError("output_dir must be provided when visualize=True")
    
    if visualize:
        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True)
    
    # Isolate protrusions
    body_mask, protrusion_masks, visualization = isolate_protrusions(mask)
    
    # Save segmentation visualization
    if visualize:
        cv2.imwrite(str(output_dir / "protrusion_segmentation.png"), visualization)
    
    # Analyze each protrusion
    protrusion_metrics = []
    
    for i, p_mask in enumerate(protrusion_masks):
        metrics = analyze_protrusion(p_mask, body_mask)
        protrusion_metrics.append(metrics)
        
        if visualize:
            # Create an RGB visualization of this protrusion
            viz = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)
            viz[:, :, 0] = body_mask * 255  # Body in blue
            viz[:, :, 1] = p_mask * 255     # Protrusion in green
            
            # Save protrusion visualization
            cv2.imwrite(str(output_dir / f"protrusion_{i+1}.png"), viz)
    
    # Sort protrusions by angle
    protrusion_metrics.sort(key=lambda x: x['angle'])
    
    if visualize:
        # Create a detailed visualization with numbered protrusions
        viz_detailed = visualization.copy()
        
        # Get body center
        body_contours, _ = cv2.findContours(
            body_mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        body_contour = max(body_contours, key=cv2.contourArea)
        M = cv2.moments(body_contour)
        if M["m00"] != 0:
            body_center_x = int(M["m10"] / M["m00"])
            body_center_y = int(M["m01"] / M["m00"])
        else:
            body_center_x, body_center_y = body_mask.shape[1]//2, body_mask.shape[0]//2
        
        # Draw numbers for each protrusion
        for i, metrics in enumerate(protrusion_metrics):
            angle_rad = np.radians(metrics['angle'])
            distance = metrics['distance_from_center'] * 0.7  # Position number at 70% of distance
            
            # Calculate position for the number
            x = int(body_center_x + distance * np.cos(angle_rad))
            y = int(body_center_y + distance * np.sin(angle_rad))
            
            # Draw number
            cv2.putText(
                viz_detailed, str(i+1), (x, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2
            )
        
        # Save detailed visualization
        cv2.imwrite(str(output_dir / "protrusions_numbered.png"), viz_detailed)
        
        # Create a plot of protrusion lengths
        plt.figure(figsize=(10, 6))
        indices = np.arange(1, len(protrusion_metrics)+1)
        lengths = [m['length'] for m in protrusion_metrics]
        
        plt.bar(indices, lengths)
        plt.xlabel('Protrusion Number')
        plt.ylabel('Length (pixels)')
        plt.title('Protrusion Lengths')
        plt.xticks(indices)
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        
        plt.tight_layout()
        plt.savefig(str(output_dir / "protrusion_lengths.png"))
        plt.close()
        
        # Create a polar plot of protrusions by angle and length
        plt.figure(figsize=(8, 8))
        angles = [np.radians(m['angle']) for m in protrusion_metrics]
        lengths = [m['length'] for m in protrusion_metrics]
        
        ax = plt.subplot(111, projection='polar')
        ax.scatter(angles, lengths, s=100, c=np.arange(len(angles)), cmap='hsv')
        
        for i, (angle, length) in enumerate(zip(angles, lengths)):
            ax.text(angle, length*1.1, str(i+1), fontsize=10, ha='center')
        
        ax.set_title('Protrusions by Angle and Length')
        
        plt.tight_layout()
        plt.savefig(str(output_dir / "protrusion_polar_plot.png"))
        plt.close()
    
    return {
        'protrusion_count': len(protrusion_masks),
        'protrusions': protrusion_metrics
    }


def summarize_protrusions(protrusion_data: Dict) -> Dict[str, float]:
    """
    Summarize protrusion metrics from a detailed analysis.
    
    Parameters:
    -----------
    protrusion_data : dict
        Output from analyze_all_protrusions
        
    Returns:
    --------
    dict
        Dictionary with summary statistics
    """
    protrusions = protrusion_data['protrusions']
    
    if not protrusions:
        return {
            'protrusion_count': 0,
            'mean_length': 0,
            'mean_width': 0,
            'mean_aspect_ratio': 0,
            'mean_area': 0,
            'length_cv': 0,  # Coefficient of variation (std/mean)
            'spacing_uniformity': 0
        }
    
    # Calculate mean values
    mean_length = np.mean([p['length'] for p in protrusions])
    mean_width = np.mean([p['width'] for p in protrusions])
    mean_aspect_ratio = np.mean([p['aspect_ratio'] for p in protrusions])
    mean_area = np.mean([p['area'] for p in protrusions])
    
    # Calculate length variability (coefficient of variation)
    lengths = np.array([p['length'] for p in protrusions])
    length_cv = np.std(lengths) / mean_length if mean_length > 0 else 0
    
    # Calculate angular spacing uniformity
    angles = np.array([p['angle'] for p in protrusions])
    
    if len(angles) >= 2:
        # Sort angles
        angles = np.sort(angles)
        
        # Calculate angular differences (including wrap-around)
        diffs = np.diff(angles)
        diffs = np.append(diffs, 360 + angles[0] - angles[-1])
        
        # For uniform spacing, all differences should be equal
        # Calculate coefficient of variation as a measure of uniformity
        angle_cv = np.std(diffs) / np.mean(diffs) if np.mean(diffs) > 0 else 0
        
        # Convert to a uniformity score (0-1)
        # 0 = completely non-uniform, 1 = perfectly uniform
        spacing_uniformity = 1 - min(angle_cv, 1.0)
    else:
        spacing_uniformity = 0
    
    return {
        'protrusion_count': len(protrusions),
        'mean_length': mean_length,
        'mean_width': mean_width,
        'mean_aspect_ratio': mean_aspect_ratio,
        'mean_area': mean_area,
        'length_cv': length_cv,
        'spacing_uniformity': spacing_uniformity
    }


if __name__ == "__main__":
    # Test code
    from seg_ana.core.synthetic import create_shape_with_protrusions
    
    # Create a test shape
    test_shape = create_shape_with_protrusions(
        size=(512, 512), radius=100, num_protrusions=6, protrusion_size=30
    )
    
    # Analyze protrusions
    results = analyze_all_protrusions(
        test_shape, visualize=True, output_dir="./protrusion_analysis_test"
    )
    
    # Print results
    print(f"Detected {results['protrusion_count']} protrusions")
    print("\nDetailed metrics:")
    for i, p in enumerate(results['protrusions']):
        print(f"Protrusion {i+1}:")
        print(f"  Length: {p['length']:.1f} pixels")
        print(f"  Width: {p['width']:.1f} pixels")
        print(f"  Aspect ratio: {p['aspect_ratio']:.2f}")
        print(f"  Angle: {p['angle']:.1f} degrees")
    
    # Print summary
    summary = summarize_protrusions(results)
    print("\nSummary statistics:")
    print(f"Protrusion count: {summary['protrusion_count']}")
    print(f"Mean length: {summary['mean_length']:.1f} pixels")
    print(f"Length coefficient of variation: {summary['length_cv']:.3f}")
    print(f"Spacing uniformity: {summary['spacing_uniformity']:.3f}")
