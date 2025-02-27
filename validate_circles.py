"""
Script to validate circle metrics and visualize the results.
"""
import os
import numpy as np
from pathlib import Path

from seg_ana.core.synthetic import create_circle_mask, create_ellipse_mask
from seg_ana.core.metrics import calculate_all_metrics
from seg_ana.core.validation import (
    export_mask_visualization,
    export_mask_with_analyzed_contour,
    compare_circle_methods,
    create_mathematical_circle,
    save_mask_to_file
)

# Create output directory
output_dir = Path("./validation_output")
output_dir.mkdir(exist_ok=True)

def main():
    print("Validating circle metrics and visualization...")
    
    # 1. Test basic circle metrics
    print("\n1. Testing basic circle metrics")
    circle_mask = create_circle_mask(size=(256, 256), radius=50, noise=0.0)
    metrics = calculate_all_metrics(circle_mask)
    
    print(f"Circle metrics using OpenCV drawing:")
    print(f"  Roundness:    {metrics['roundness']:.4f} (should be close to 1.0)")
    print(f"  Ellipticity:  {metrics['ellipticity']:.4f} (should be close to 1.0)")
    print(f"  Solidity:     {metrics['solidity']:.4f} (should be close to 1.0)")
    
    # Export visualization
    viz_path = export_mask_visualization(
        circle_mask, 
        output_dir / "circle_viz.png"
    )
    print(f"  Basic visualization saved to: {viz_path}")
    
    # Export detailed analysis
    analysis_path = export_mask_with_analyzed_contour(
        circle_mask,
        output_dir / "circle_analysis.png"
    )
    print(f"  Detailed analysis saved to: {analysis_path}")
    
    # 2. Compare different methods of creating circles
    print("\n2. Comparing different circle creation methods")
    results = compare_circle_methods(
        radius=50, 
        size=(256, 256),
        save_path=output_dir / "circle_comparison"
    )
    
    print(f"  OpenCV circle roundness:      {results['opencv']:.4f}")
    print(f"  Mathematical circle roundness: {results['mathematical']:.4f}")
    
    # 3. Test with different radii
    print("\n3. Testing roundness with different circle radii")
    radii = [20, 50, 100]
    
    for radius in radii:
        # Create circles with both methods
        cv_circle = create_circle_mask(size=(256, 256), radius=radius)
        math_circle = create_mathematical_circle(size=(256, 256), radius=radius)
        
        # Calculate metrics
        cv_metrics = calculate_all_metrics(cv_circle)
        math_metrics = calculate_all_metrics(math_circle)
        
        print(f"  Radius {radius}:")
        print(f"    OpenCV roundness:      {cv_metrics['roundness']:.4f}")
        print(f"    Mathematical roundness: {math_metrics['roundness']:.4f}")
    
    # 4. Test if exporting the mask helps for visual inspection
    print("\n4. Exporting masks for visual inspection")
    
    # Export OpenCV circle
    cv_npy, cv_png = save_mask_to_file(
        circle_mask,
        output_dir / "opencv_circle"
    )
    print(f"  OpenCV circle mask saved to: {cv_npy}")
    
    # Export Mathematical circle
    math_circle = create_mathematical_circle(size=(256, 256), radius=50)
    math_npy, math_png = save_mask_to_file(
        math_circle,
        output_dir / "mathematical_circle"
    )
    print(f"  Mathematical circle mask saved to: {math_npy}")
    
    print("\nValidation complete. Check the output directory for results.")
    print(f"Output directory: {output_dir.absolute()}")

if __name__ == "__main__":
    main()
