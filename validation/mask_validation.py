"""
Script for validating and visualizing synthetic masks and their metrics.
"""
import os
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from seg_ana.core.synthetic import (
    create_circle_mask, 
    create_ellipse_mask, 
    create_shape_with_protrusions
)
from seg_ana.core.metrics import calculate_all_metrics, get_largest_contour
from seg_ana.visualization.basic import (
    visualize_mask,
    visualize_mask_with_metrics
)

# Create output directory if it doesn't exist
output_dir = Path("./validation_output")
output_dir.mkdir(exist_ok=True)

def save_mask_to_image(mask, filename):
    """Save a binary mask as an image file."""
    plt.figure(figsize=(8, 8))
    plt.imshow(mask, cmap='gray')
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    plt.close()

def save_mask_with_contour(mask, filename):
    """Save a binary mask with its contour overlay."""
    plt.figure(figsize=(8, 8))
    plt.imshow(mask, cmap='gray')
    
    # Draw contour
    contour = get_largest_contour(mask.astype(np.uint8))
    if contour.size > 0:
        contour_reshaped = contour.squeeze()
        if contour_reshaped.ndim == 2:  # Normal case
            plt.plot(contour_reshaped[:, 0], contour_reshaped[:, 1], 'r-', linewidth=2)
    
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    plt.close()

def validate_circle():
    """Validate a perfect circle."""
    print("\n=== Validating Perfect Circle ===")
    
    # Create perfect circle
    circle_mask = create_circle_mask(size=(256, 256), radius=50, noise=0.0)
    
    # Calculate metrics
    metrics = calculate_all_metrics(circle_mask)
    
    # Print metrics
    print(f"Circle metrics:")
    print(f"  Roundness:    {metrics['roundness']:.4f} (should be close to 1.0)")
    print(f"  Ellipticity:  {metrics['ellipticity']:.4f} (should be close to 1.0)")
    print(f"  Solidity:     {metrics['solidity']:.4f} (should be close to 1.0)")
    print(f"  Perimeter:    {metrics['perimeter']:.2f}")
    print(f"  Area:         {metrics['area']:.2f}")
    
    # Save mask and visualization
    save_mask_to_image(circle_mask, output_dir / "circle_mask.png")
    save_mask_with_contour(circle_mask, output_dir / "circle_with_contour.png")
    
    # Save mask with metrics
    fig = visualize_mask_with_metrics(circle_mask, metrics)
    fig.savefig(output_dir / "circle_with_metrics.png", dpi=150, bbox_inches='tight')
    plt.close(fig)
    
    # Save raw mask data
    np.save(output_dir / "circle_mask.npy", circle_mask)

def validate_ellipse():
    """Validate ellipses with different axis ratios."""
    print("\n=== Validating Ellipses ===")
    
    # Create ellipses with different axis ratios
    ratios = [1.0, 2.0, 3.0]
    
    for ratio in ratios:
        major_axis = 60
        minor_axis = int(major_axis / ratio)
        
        print(f"\nEllipse with {ratio}:1 axis ratio:")
        
        # Create ellipse
        ellipse_mask = create_ellipse_mask(
            size=(256, 256), 
            axes=(major_axis, minor_axis)
        )
        
        # Calculate metrics
        metrics = calculate_all_metrics(ellipse_mask)
        
        # Print metrics
        print(f"  Ellipticity:  {metrics['ellipticity']:.4f} (should be close to {ratio})")
        print(f"  Roundness:    {metrics['roundness']:.4f}")
        print(f"  Solidity:     {metrics['solidity']:.4f}")
        
        # Save mask and visualization
        save_mask_to_image(
            ellipse_mask, 
            output_dir / f"ellipse_{ratio}_1_mask.png"
        )
        save_mask_with_contour(
            ellipse_mask, 
            output_dir / f"ellipse_{ratio}_1_contour.png"
        )
        
        # Save mask with metrics
        fig = visualize_mask_with_metrics(ellipse_mask, metrics)
        fig.savefig(
            output_dir / f"ellipse_{ratio}_1_metrics.png", 
            dpi=150, 
            bbox_inches='tight'
        )
        plt.close(fig)
        
        # Save raw mask data
        np.save(output_dir / f"ellipse_{ratio}_1_mask.npy", ellipse_mask)

def validate_protrusions():
    """Validate shapes with protrusions."""
    print("\n=== Validating Shapes with Protrusions ===")
    
    # Create shapes with different numbers of protrusions
    protrusion_counts = [0, 3, 6]
    
    for count in protrusion_counts:
        print(f"\nShape with {count} protrusions:")
        
        # Create shape
        shape_mask = create_shape_with_protrusions(
            size=(256, 256),
            radius=50,
            num_protrusions=count,
            protrusion_size=10,
            protrusion_distance=1.3
        )
        
        # Calculate metrics
        metrics = calculate_all_metrics(shape_mask)
        
        # Print metrics
        print(f"  Detected protrusions: {metrics['protrusions']}")
        print(f"  Roundness:            {metrics['roundness']:.4f}")
        print(f"  Solidity:             {metrics['solidity']:.4f}")
        
        # Save mask and visualization
        save_mask_to_image(
            shape_mask, 
            output_dir / f"protrusions_{count}_mask.png"
        )
        save_mask_with_contour(
            shape_mask, 
            output_dir / f"protrusions_{count}_contour.png"
        )
        
        # Save mask with metrics
        fig = visualize_mask_with_metrics(shape_mask, metrics)
        fig.savefig(
            output_dir / f"protrusions_{count}_metrics.png", 
            dpi=150, 
            bbox_inches='tight'
        )
        plt.close(fig)
        
        # Save raw mask data
        np.save(output_dir / f"protrusions_{count}_mask.npy", shape_mask)

def investigate_roundness():
    """Investigate roundness calculation with analytical approach."""
    print("\n=== Investigating Roundness Calculation ===")
    
    # Create a perfect circle with various radii
    radii = [25, 50, 100]
    
    for radius in radii:
        print(f"\nCircle with radius {radius}:")
        
        # Create circle
        circle_mask = create_circle_mask(size=(512, 512), radius=radius, noise=0.0)
        
        # Calculate metrics
        metrics = calculate_all_metrics(circle_mask)
        
        # Get contour for analytical calculation
        contour = get_largest_contour(circle_mask.astype(np.uint8))
        perimeter = cv2.arcLength(contour, True)
        area = cv2.contourArea(contour)
        
        # Theoretical values
        theoretical_area = np.pi * radius**2
        theoretical_perimeter = 2 * np.pi * radius
        theoretical_roundness = 1.0
        
        # Print comparison
        print(f"  Measured area:      {area:.2f}")
        print(f"  Theoretical area:   {theoretical_area:.2f}")
        print(f"  Difference:         {(area - theoretical_area) / theoretical_area * 100:.2f}%")
        print(f"  Measured perimeter: {perimeter:.2f}")
        print(f"  Theoretical perim:  {theoretical_perimeter:.2f}")
        print(f"  Difference:         {(perimeter - theoretical_perimeter) / theoretical_perimeter * 100:.2f}%")
        print(f"  Calculated roundness: {metrics['roundness']:.4f}")
        print(f"  Manual roundness:     {4 * np.pi * area / (perimeter**2):.4f}")
        print(f"  Theoretical roundness: {theoretical_roundness:.4f}")
        
        # Save contour points for inspection
        if radius == 50:  # Save one example
            contour_points = contour.squeeze()
            np.savetxt(
                output_dir / "contour_points.csv", 
                contour_points, 
                delimiter=",", 
                header="x,y", 
                comments=""
            )

if __name__ == "__main__":
    # Create output directory
    print(f"Saving output to {output_dir.absolute()}")
    
    # Run validations
    validate_circle()
    validate_ellipse()
    validate_protrusions()
    
    try:
        import cv2
        investigate_roundness()
    except ImportError:
        print("OpenCV required for roundness investigation")
    
    print("\nValidation complete. Check the output directory for results.")
