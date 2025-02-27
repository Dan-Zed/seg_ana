"""
Basic usage example for the seg-ana package.

This example demonstrates how to:
1. Generate synthetic test shapes
2. Calculate basic metrics
3. Use enhanced protrusion detection
4. Visualize the results
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Import from seg_ana package
from seg_ana.core.synthetic import (
    create_shape_with_protrusions,
    create_mathematical_circle
)
from seg_ana.core.metrics_improved import calculate_all_metrics
from seg_ana.core.protrusion_analysis import analyze_all_protrusions

# Output directory
output_dir = Path("./example_output")
output_dir.mkdir(exist_ok=True)


def analyze_and_plot(shape, title, filename):
    """Analyze a shape and create a visualization."""
    # Calculate metrics
    metrics = calculate_all_metrics(shape)
    
    # Enhanced protrusion analysis
    protrusion_results = analyze_all_protrusions(shape)
    
    # Create a visualization
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
    
    # Original shape
    ax1.imshow(shape, cmap='gray')
    ax1.set_title("Original Shape")
    ax1.axis('off')
    
    # Get contour for visualization
    import cv2
    contours, _ = cv2.findContours(
        shape.astype(np.uint8), 
        cv2.RETR_EXTERNAL, 
        cv2.CHAIN_APPROX_SIMPLE
    )
    if contours:
        contour = max(contours, key=cv2.contourArea)
        hull = cv2.convexHull(contour)
        
        # Overlay contour
        contour_points = contour.squeeze()
        hull_points = hull.squeeze()
        
        ax1.plot(contour_points[:, 0], contour_points[:, 1], 'r-', linewidth=2)
        ax1.plot(hull_points[:, 0], hull_points[:, 1], 'g-', linewidth=2)
    
    # Metrics text
    metrics_text = (
        f"Area: {metrics.get('area', 'N/A'):.1f} pixels²\n"
        f"Perimeter: {metrics.get('perimeter', 'N/A'):.1f} pixels\n"
        f"Roundness: {metrics.get('roundness', 'N/A'):.4f}\n"
        f"Ellipticity: {metrics.get('ellipticity', 'N/A'):.4f}\n"
        f"Solidity: {metrics.get('solidity', 'N/A'):.4f}\n"
        f"Protrusions: {metrics.get('protrusions', 'N/A')}\n"
    )
    
    # Add enhanced protrusion metrics if available
    if 'protrusion_mean_length' in metrics:
        metrics_text += (
            f"Mean Protrusion Length: {metrics.get('protrusion_mean_length', 'N/A'):.1f} pixels\n"
            f"Mean Protrusion Width: {metrics.get('protrusion_mean_width', 'N/A'):.1f} pixels\n"
            f"Length Variation (CV): {metrics.get('protrusion_length_cv', 'N/A'):.3f}\n"
            f"Spacing Uniformity: {metrics.get('protrusion_spacing_uniformity', 'N/A'):.3f}\n"
        )
    
    # Display metrics
    ax2.text(0.1, 0.5, metrics_text, transform=ax2.transAxes, fontsize=12,
             verticalalignment='center', fontfamily='monospace')
    ax2.axis('off')
    
    # Set title
    fig.suptitle(title, fontsize=16)
    
    # Save the figure
    plt.tight_layout()
    plt.savefig(output_dir / filename, dpi=150)
    plt.close()
    
    return metrics


def main():
    """Run the example analysis on synthetic shapes."""
    print("Running basic analysis example...")
    
    # 1. Create a perfect circle
    circle = create_mathematical_circle(size=(512, 512), radius=100)
    circle_metrics = analyze_and_plot(
        circle, "Perfect Circle", "circle_analysis.png"
    )
    print(f"Circle analysis complete. Protrusions: {circle_metrics['protrusions']}")
    
    # 2. Create a shape with protrusions
    shape_with_protrusions = create_shape_with_protrusions(
        size=(512, 512),
        radius=100,
        num_protrusions=6,
        protrusion_size=20
    )
    protrusion_metrics = analyze_and_plot(
        shape_with_protrusions, "Shape with 6 Protrusions", "protrusions_analysis.png"
    )
    print(f"Protrusion analysis complete. Detected: {protrusion_metrics['protrusions']}")
    
    # 3. Create an ellipse with a higher axis ratio
    from seg_ana.core.synthetic import create_ellipse_mask
    ellipse = create_ellipse_mask(
        size=(512, 512),
        axes=(150, 75)
    )
    ellipse_metrics = analyze_and_plot(
        ellipse, "Ellipse (2:1 ratio)", "ellipse_analysis.png"
    )
    print(f"Ellipse analysis complete. Ellipticity: {ellipse_metrics['ellipticity']:.4f}")
    
    print(f"All analyses complete. Results saved to {output_dir.absolute()}")


if __name__ == "__main__":
    main()
