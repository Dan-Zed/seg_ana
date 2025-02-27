"""
Utility script to save mask visualizations and data.

This script generates and saves various types of masks (circle, ellipse, etc.)
for visual inspection and debugging.
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
from seg_ana.core.metrics_improved import (
    calculate_all_metrics,
    create_mathematical_circle
)

# Create output directory
output_dir = Path("./mask_exports")
output_dir.mkdir(exist_ok=True)

def save_mask_image(mask, filename, title=None):
    """Save a mask as a PNG image."""
    plt.figure(figsize=(8, 8))
    plt.imshow(mask, cmap='gray')
    if title:
        plt.title(title)
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    plt.close()

def main():
    print(f"Saving mask visualizations to: {output_dir.absolute()}")
    
    # 1. Create and save circle masks
    print("\n1. Saving circle masks...")
    
    # OpenCV circle
    cv_circle = create_circle_mask(size=(256, 256), radius=50)
    np.save(output_dir / "circle_opencv.npy", cv_circle)
    save_mask_image(
        cv_circle, 
        output_dir / "circle_opencv.png",
        "OpenCV Circle (r=50)"
    )
    
    # Mathematical circle
    math_circle = create_mathematical_circle(size=(256, 256), radius=50)
    np.save(output_dir / "circle_mathematical.npy", math_circle)
    save_mask_image(
        math_circle, 
        output_dir / "circle_mathematical.png",
        "Mathematical Circle (r=50)"
    )
    
    metrics_cv = calculate_all_metrics(cv_circle)
    metrics_math = calculate_all_metrics(math_circle)
    
    print(f"  OpenCV Circle Metrics:")
    print(f"    Roundness:    {metrics_cv['roundness']:.4f}")
    print(f"    Ellipticity:  {metrics_cv['ellipticity']:.4f}")
    print(f"    Solidity:     {metrics_cv['solidity']:.4f}")
    
    print(f"  Mathematical Circle Metrics:")
    print(f"    Roundness:    {metrics_math['roundness']:.4f}")
    print(f"    Ellipticity:  {metrics_math['ellipticity']:.4f}")
    print(f"    Solidity:     {metrics_math['solidity']:.4f}")
    
    # 2. Create and save ellipse masks
    print("\n2. Saving ellipse masks...")
    
    # Various axis ratios
    ratios = [1.0, 2.0, 3.0]
    
    for ratio in ratios:
        major_axis = 60
        minor_axis = int(major_axis / ratio)
        
        ellipse_mask = create_ellipse_mask(
            size=(256, 256), 
            axes=(major_axis, minor_axis)
        )
        
        np.save(output_dir / f"ellipse_{ratio}_1.npy", ellipse_mask)
        save_mask_image(
            ellipse_mask,
            output_dir / f"ellipse_{ratio}_1.png",
            f"Ellipse {ratio}:1 axis ratio"
        )
        
        metrics = calculate_all_metrics(ellipse_mask)
        
        print(f"  Ellipse {ratio}:1 Metrics:")
        print(f"    Ellipticity:  {metrics['ellipticity']:.4f} (expected: {ratio:.1f})")
        print(f"    Roundness:    {metrics['roundness']:.4f}")
        print(f"    Solidity:     {metrics['solidity']:.4f}")
    
    # 3. Create and save shapes with protrusions
    print("\n3. Saving shapes with protrusions...")
    
    protrusion_counts = [0, 3, 6]
    
    for count in protrusion_counts:
        shape_mask = create_shape_with_protrusions(
            size=(256, 256),
            radius=50,
            num_protrusions=count,
            protrusion_size=10,
            protrusion_distance=1.3
        )
        
        np.save(output_dir / f"protrusions_{count}.npy", shape_mask)
        save_mask_image(
            shape_mask,
            output_dir / f"protrusions_{count}.png",
            f"Shape with {count} protrusions"
        )
        
        metrics = calculate_all_metrics(shape_mask)
        
        print(f"  Shape with {count} protrusions Metrics:")
        print(f"    Detected protrusions: {metrics['protrusions']}")
        print(f"    Roundness:            {metrics['roundness']:.4f}")
        print(f"    Solidity:             {metrics['solidity']:.4f}")
    
    # 4. Create and save circle with noise
    print("\n4. Saving circle with noise...")
    
    noise_levels = [0.0, 0.1, 0.2, 0.3]
    
    for noise in noise_levels:
        noisy_circle = create_circle_mask(
            size=(256, 256),
            radius=50,
            noise=noise
        )
        
        np.save(output_dir / f"circle_noise_{noise:.1f}.npy", noisy_circle)
        save_mask_image(
            noisy_circle,
            output_dir / f"circle_noise_{noise:.1f}.png",
            f"Circle with noise level {noise:.1f}"
        )
        
        metrics = calculate_all_metrics(noisy_circle)
        
        print(f"  Circle with noise {noise:.1f} Metrics:")
        print(f"    Roundness:    {metrics['roundness']:.4f}")
        print(f"    Ellipticity:  {metrics['ellipticity']:.4f}")
        print(f"    Solidity:     {metrics['solidity']:.4f}")
    
    # Create a summary page
    print("\n5. Creating summary visualization...")
    
    # Arrange masks in a grid
    fig, axes = plt.subplots(3, 4, figsize=(16, 12))
    axes = axes.flatten()
    
    # Add OpenCV and mathematical circles
    axes[0].imshow(cv_circle, cmap='gray')
    axes[0].set_title(f"OpenCV Circle\nRoundness: {metrics_cv['roundness']:.4f}")
    axes[0].axis('off')
    
    axes[1].imshow(math_circle, cmap='gray')
    axes[1].set_title(f"Mathematical Circle\nRoundness: {metrics_math['roundness']:.4f}")
    axes[1].axis('off')
    
    # Add ellipses
    for i, ratio in enumerate(ratios):
        idx = i + 2
        ellipse_mask = np.load(output_dir / f"ellipse_{ratio}_1.npy")
        metrics = calculate_all_metrics(ellipse_mask)
        
        axes[idx].imshow(ellipse_mask, cmap='gray')
        axes[idx].set_title(f"Ellipse {ratio}:1\nEllipticity: {metrics['ellipticity']:.2f}")
        axes[idx].axis('off')
    
    # Add protrusion shapes
    for i, count in enumerate(protrusion_counts):
        idx = i + 5
        protrusion_mask = np.load(output_dir / f"protrusions_{count}.npy")
        metrics = calculate_all_metrics(protrusion_mask)
        
        axes[idx].imshow(protrusion_mask, cmap='gray')
        axes[idx].set_title(f"{count} Protrusions\nSolidity: {metrics['solidity']:.2f}")
        axes[idx].axis('off')
    
    # Add noisy circles
    for i, noise in enumerate(noise_levels):
        idx = i + 8
        noisy_mask = np.load(output_dir / f"circle_noise_{noise:.1f}.npy")
        metrics = calculate_all_metrics(noisy_mask)
        
        axes[idx].imshow(noisy_mask, cmap='gray')
        axes[idx].set_title(f"Noise: {noise:.1f}\nRoundness: {metrics['roundness']:.2f}")
        axes[idx].axis('off')
    
    plt.tight_layout()
    plt.savefig(output_dir / "shape_summary.png", dpi=200, bbox_inches='tight')
    plt.close()
    
    print(f"\nAll masks saved to: {output_dir.absolute()}")
    print(f"Summary visualization: {output_dir / 'shape_summary.png'}")
    
if __name__ == "__main__":
    main()
