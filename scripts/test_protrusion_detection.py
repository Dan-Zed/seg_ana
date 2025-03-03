"""
Script to test and validate the improved protrusion detection algorithm.
"""
import os
import numpy as np
import matplotlib.pyplot as plt
import cv2
from pathlib import Path

from seg_ana.core.synthetic import create_shape_with_protrusions
from seg_ana.core.metrics import calculate_all_metrics

# Create output directory
output_dir = Path("./protrusion_test")
output_dir.mkdir(exist_ok=True)


def get_largest_contour(mask):
    """Get the largest contour from a binary mask."""
    contours, _ = cv2.findContours(
        mask.astype(np.uint8), 
        cv2.RETR_EXTERNAL, 
        cv2.CHAIN_APPROX_SIMPLE
    )
    if not contours:
        return np.array([])
    return max(contours, key=cv2.contourArea)


def count_protrusions_old(contour, hull=None, threshold=2.0):
    """Original protrusion counting method for comparison."""
    if hull is None:
        hull = cv2.convexHull(contour)
    
    contour_points = contour.squeeze()
    hull_points = hull.squeeze()
    
    if contour_points.ndim == 1 or hull_points.ndim == 1:
        return 0
    
    # Calculate distances
    distances = np.min(
        np.linalg.norm(contour_points[:, None] - hull_points[None], axis=2),
        axis=1
    )
    
    # Simple counting
    protrusions = np.sum(distances > threshold)
    return int(protrusions)


def count_protrusions_new(contour, hull=None, threshold=5.0):
    """New improved protrusion counting method."""
    if hull is None:
        hull = cv2.convexHull(contour)
    
    contour_points = contour.squeeze()
    hull_points = hull.squeeze()
    
    if contour_points.ndim == 1 or hull_points.ndim == 1:
        return 0
    
    # Calculate distances
    distances = np.min(
        np.linalg.norm(contour_points[:, None] - hull_points[None], axis=2),
        axis=1
    )
    
    # Improved method: group adjacent points
    # Identify potential protrusion regions
    protrusion_points = distances > threshold
    indices = np.where(protrusion_points)[0]
    
    if len(indices) == 0:
        return 0
    
    # Identify distinct groups
    distinct_groups = []
    current_group = [indices[0]]
    
    for i in range(1, len(indices)):
        prev_idx = indices[i-1]
        curr_idx = indices[i]
        contour_length = len(contour_points)
        
        # Distance with wrap-around
        dist = min(abs(curr_idx - prev_idx), contour_length - abs(curr_idx - prev_idx))
        
        if dist <= 5:  # Points close together are part of the same protrusion
            current_group.append(curr_idx)
        else:
            distinct_groups.append(current_group)
            current_group = [curr_idx]
    
    # Add the last group
    if current_group:
        distinct_groups.append(current_group)
    
    return len(distinct_groups)


def visualize_protrusions(mask, original_count, new_count, filename):
    """Create a visualization of protrusion detection."""
    # Get contour and hull
    contour = get_largest_contour(mask)
    hull = cv2.convexHull(contour)
    
    # Create a visualization image (RGB)
    viz = np.stack([mask * 255] * 3, axis=2).astype(np.uint8)
    
    # Draw the convex hull in green
    cv2.drawContours(viz, [hull], 0, (0, 255, 0), 2)
    
    # Draw the original contour in blue
    cv2.drawContours(viz, [contour], 0, (255, 0, 0), 2)
    
    # Add text with counts
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(viz, f"Old method: {original_count} protrusions", 
                (20, 30), font, 0.8, (0, 0, 255), 2)
    cv2.putText(viz, f"New method: {new_count} protrusions", 
                (20, 70), font, 0.8, (255, 0, 0), 2)
    
    # Save the visualization
    cv2.imwrite(str(output_dir / filename), viz)
    
    return viz


def main():
    print(f"Testing protrusion detection improvements")
    print(f"Results will be saved to: {output_dir.absolute()}")
    
    # Test with different numbers of protrusions
    protrusion_counts = [0, 3, 6, 9]
    results = []
    
    # Create a figure to display all results
    fig, axes = plt.subplots(len(protrusion_counts), 1, figsize=(10, 5*len(protrusion_counts)))
    
    for i, count in enumerate(protrusion_counts):
        print(f"\nTesting with {count} protrusions...")
        
        # Create shape with protrusions
        mask = create_shape_with_protrusions(
            size=(512, 512),
            radius=100,
            num_protrusions=count,
            protrusion_size=20,
            protrusion_distance=1.3
        )
        
        # Save the original mask
        np.save(output_dir / f"shape_with_{count}_protrusions.npy", mask)
        
        # Get contour and hull
        contour = get_largest_contour(mask)
        hull = cv2.convexHull(contour)
        
        # Calculate protrusions with old and new methods
        old_count = count_protrusions_old(contour, hull)
        new_count = count_protrusions_new(contour, hull)
        
        # Visualize the results
        viz = visualize_protrusions(
            mask, 
            old_count, 
            new_count, 
            f"protrusions_{count}_comparison.png"
        )
        
        # Store results
        results.append({
            'created_protrusions': count,
            'old_method_count': old_count,
            'new_method_count': new_count
        })
        
        # Display in the figure
        if isinstance(axes, np.ndarray):
            ax = axes[i]
        else:
            ax = axes  # For the case of only one subplot
        
        ax.imshow(viz)
        ax.set_title(f"{count} Protrusions: Old method={old_count}, New method={new_count}")
        ax.axis('off')
    
    # Save the figure
    plt.tight_layout()
    plt.savefig(output_dir / "protrusion_detection_comparison.png", dpi=150)
    
    # Print summary table
    print("\nProtrusion Detection Comparison:")
    print("----------------------------------------")
    print("Created | Old Method Count | New Method Count")
    print("----------------------------------------")
    for result in results:
        print(f"   {result['created_protrusions']}    |        {result['old_method_count']}        |        {result['new_method_count']}")
    
    # Also test with a noisy circle (which should have 0 protrusions)
    print("\nTesting with a noisy circle...")
    from seg_ana.core.synthetic import create_circle_mask
    
    noisy_circle = create_circle_mask(
        size=(512, 512),
        radius=100,
        noise=0.2
    )
    
    contour = get_largest_contour(noisy_circle)
    hull = cv2.convexHull(contour)
    
    old_noise_count = count_protrusions_old(contour, hull)
    new_noise_count = count_protrusions_new(contour, hull)
    
    visualize_protrusions(
        noisy_circle, 
        old_noise_count, 
        new_noise_count, 
        "noisy_circle_comparison.png"
    )
    
    print(f"Noisy circle (noise=0.2):")
    print(f"  Old method: {old_noise_count} protrusions")
    print(f"  New method: {new_noise_count} protrusions")
    
    print(f"\nAll tests complete. Results saved to: {output_dir.absolute()}")


if __name__ == "__main__":
    main()
