"""
Script to demonstrate the behavior of metrics with vs. without normalization.
This file replaces the old test_improved_metrics.py that compared two different implementations.

Instead, it now creates circles and ellipses with known properties and shows how the metrics 
behave when calculating their shape characteristics.
"""
import os
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from seg_ana.core.synthetic import create_circle_mask, create_ellipse_mask
from seg_ana.core.metrics import (
    calculate_all_metrics,
    create_mathematical_circle
)

# Create output directory
output_dir = Path("./metrics_test")
output_dir.mkdir(exist_ok=True)

def main():
    print("\nTesting metrics calculation for perfect shapes")
    print(f"Saving results to: {output_dir.absolute()}")
    
    # Test parameters
    test_radii = [20, 50, 100]
    
    # Create a table for results
    results = []
    
    # Test each radius
    for radius in test_radii:
        print(f"\nTesting circles with radius {radius}:")
        
        # Create circles using different methods
        opencv_circle = create_circle_mask(size=(256, 256), radius=radius, noise=0.0)
        math_circle = create_mathematical_circle(size=(256, 256), radius=radius)
        
        # Calculate metrics using both methods
        opencv_metrics = calculate_all_metrics(opencv_circle)
        math_metrics = calculate_all_metrics(math_circle)
        
        # Print results
        print(f"  1. OpenCV Circle:")
        print(f"     Roundness: {opencv_metrics['roundness']:.4f}")
        print(f"     Ellipticity: {opencv_metrics['ellipticity']:.4f}")
        print(f"     Solidity: {opencv_metrics['solidity']:.4f}")
        
        print(f"  2. Mathematical Circle:")
        print(f"     Roundness: {math_metrics['roundness']:.4f}")
        print(f"     Ellipticity: {math_metrics['ellipticity']:.4f}")
        print(f"     Solidity: {math_metrics['solidity']:.4f}")
        
        # Store results for visualization
        results.append({
            'radius': radius,
            'opencv_roundness': opencv_metrics['roundness'],
            'math_roundness': math_metrics['roundness']
        })
        
        # Visualize masks for this radius
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
        
        # OpenCV circle
        ax1.imshow(opencv_circle, cmap='gray')
        ax1.set_title(f"OpenCV Circle\nRoundness: {opencv_metrics['roundness']:.4f}")
        ax1.axis('off')
        
        # Mathematical circle
        ax2.imshow(math_circle, cmap='gray')
        ax2.set_title(f"Mathematical Circle\nRoundness: {math_metrics['roundness']:.4f}")
        ax2.axis('off')
        
        plt.tight_layout()
        plt.savefig(output_dir / f"circle_radius_{radius}_comparison.png", dpi=150)
        plt.close()
    
    # Create comparison bar chart
    create_comparison_chart(results)
    
    # Test ellipses with different axis ratios
    test_ellipses()
    
    print("\nMetrics testing complete.")

def create_comparison_chart(results):
    """Create a bar chart comparing roundness values from different methods."""
    radii = [r['radius'] for r in results]
    
    # Set up bar positions
    bar_width = 0.3
    r1 = np.arange(len(radii))
    r2 = [x + bar_width for x in r1]
    
    # Create figure
    plt.figure(figsize=(10, 6))
    
    # Create bars
    plt.bar(r1, [r['opencv_roundness'] for r in results], 
            width=bar_width, label='OpenCV Circle', alpha=0.7)
    plt.bar(r2, [r['math_roundness'] for r in results], 
            width=bar_width, label='Mathematical Circle', alpha=0.7)
    
    # Add reference line for perfect roundness
    plt.axhline(y=1.0, color='r', linestyle='--', alpha=0.5, label='Perfect Circle (1.0)')
    
    # Customize chart
    plt.xlabel('Circle Radius')
    plt.ylabel('Roundness')
    plt.title('Comparison of Roundness Calculations')
    plt.xticks([r + bar_width/2 for r in range(len(radii))], radii)
    plt.ylim(0.9, 1.05)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / "roundness_comparison_chart.png", dpi=150)
    plt.close()

def test_ellipses():
    """Test metrics with ellipses of different axis ratios."""
    print("\nTesting ellipses with different axis ratios:")
    
    # Test ellipses with different axis ratios
    ratios = [1.0, 2.0, 3.0, 4.0]
    
    # Create a table for results
    ellipse_results = []
    
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
        
        # Print results
        print(f"  Ellipticity: {metrics['ellipticity']:.4f} (expected: {1/ratio:.4f})")
        print(f"  Roundness: {metrics['roundness']:.4f}")
        
        # Store results
        ellipse_results.append({
            'ratio': ratio,
            'ellipticity': metrics['ellipticity'],
            'expected_ellipticity': 1/ratio,  # minor/major
            'roundness': metrics['roundness'],
            'expected_roundness': 4*ratio/(1+ratio)**2  # Theoretical roundness for an ellipse
        })
        
        # Create visualization
        plt.figure(figsize=(8, 8))
        plt.imshow(ellipse_mask, cmap='gray')
        plt.title(f"Ellipse {ratio}:1\nEllipticity: {metrics['ellipticity']:.4f}, Roundness: {metrics['roundness']:.4f}")
        plt.axis('off')
        
        plt.tight_layout()
        plt.savefig(output_dir / f"ellipse_ratio_{ratio}.png", dpi=150)
        plt.close()
    
    # Create comparison chart for ellipses
    create_ellipse_comparison_chart(ellipse_results)

def create_ellipse_comparison_chart(results):
    """Create chart comparing ellipticity and roundness for ellipses."""
    ratios = [r['ratio'] for r in results]
    
    # Set up figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Plot ellipticity chart
    ax1.plot(ratios, [r['expected_ellipticity'] for r in results], 'r--', alpha=0.7, label='Expected (minor/major)')
    ax1.plot(ratios, [r['ellipticity'] for r in results], 'bo-', alpha=0.7, label='Measured')
    
    ax1.set_xlabel('Major/Minor Axis Ratio')
    ax1.set_ylabel('Ellipticity (minor/major)')
    ax1.set_title('Ellipticity Measurements')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # Plot roundness chart
    ax2.plot(ratios, [r['expected_roundness'] for r in results], 'r--', alpha=0.7, label='Theoretical')
    ax2.plot(ratios, [r['roundness'] for r in results], 'bo-', alpha=0.7, label='Measured')
    
    ax2.set_xlabel('Axis Ratio')
    ax2.set_ylabel('Roundness')
    ax2.set_title('Roundness Measurements')
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    
    plt.tight_layout()
    plt.savefig(output_dir / "ellipse_metrics_comparison.png", dpi=150)
    plt.close()

if __name__ == "__main__":
    main()
