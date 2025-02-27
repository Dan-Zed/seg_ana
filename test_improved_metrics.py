"""
Script to test the improved metrics calculation for perfect shapes.
"""
import os
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from seg_ana.core.synthetic import create_circle_mask, create_ellipse_mask
from seg_ana.core.metrics import calculate_all_metrics as original_metrics
from seg_ana.core.metrics_improved import (
    calculate_all_metrics as improved_metrics,
    create_mathematical_circle
)

# Create output directory
output_dir = Path("./improved_metrics_test")
output_dir.mkdir(exist_ok=True)

def main():
    print("\nTesting improved metrics calculation for perfect shapes")
    print(f"Saving results to: {output_dir.absolute()}")
    
    # Test parameters
    test_radii = [20, 50, 100]
    test_methods = [
        "OpenCV Circle (Original Metrics)",
        "OpenCV Circle (Improved Metrics)",
        "Mathematical Circle (Original Metrics)",
        "Mathematical Circle (Improved Metrics)"
    ]
    
    # Create a table for results
    results = []
    
    # Test each radius
    for radius in test_radii:
        print(f"\nTesting circles with radius {radius}:")
        
        # Create circles using different methods
        opencv_circle = create_circle_mask(size=(256, 256), radius=radius, noise=0.0)
        math_circle = create_mathematical_circle(size=(256, 256), radius=radius)
        
        # Calculate metrics using both original and improved methods
        opencv_original = original_metrics(opencv_circle)
        opencv_improved = improved_metrics(opencv_circle)
        math_original = original_metrics(math_circle)
        math_improved = improved_metrics(math_circle)
        
        # Print results
        print(f"  1. OpenCV Circle (Original Metrics):")
        print(f"     Roundness: {opencv_original['roundness']:.4f}")
        print(f"     Ellipticity: {opencv_original['ellipticity']:.4f}")
        print(f"     Solidity: {opencv_original['solidity']:.4f}")
        
        print(f"  2. OpenCV Circle (Improved Metrics):")
        print(f"     Roundness: {opencv_improved['roundness']:.4f}")
        print(f"     Ellipticity: {opencv_improved['ellipticity']:.4f}")
        print(f"     Solidity: {opencv_improved['solidity']:.4f}")
        
        print(f"  3. Mathematical Circle (Original Metrics):")
        print(f"     Roundness: {math_original['roundness']:.4f}")
        print(f"     Ellipticity: {math_original['ellipticity']:.4f}")
        print(f"     Solidity: {math_original['solidity']:.4f}")
        
        print(f"  4. Mathematical Circle (Improved Metrics):")
        print(f"     Roundness: {math_improved['roundness']:.4f}")
        print(f"     Ellipticity: {math_improved['ellipticity']:.4f}")
        print(f"     Solidity: {math_improved['solidity']:.4f}")
        
        # Store results for visualization
        results.append({
            'radius': radius,
            'opencv_original_roundness': opencv_original['roundness'],
            'opencv_improved_roundness': opencv_improved['roundness'],
            'math_original_roundness': math_original['roundness'],
            'math_improved_roundness': math_improved['roundness']
        })
        
        # Visualize masks for this radius
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 10))
        
        # OpenCV circle with original metrics
        ax1.imshow(opencv_circle, cmap='gray')
        ax1.set_title(f"OpenCV Circle (Original)\nRoundness: {opencv_original['roundness']:.4f}")
        ax1.axis('off')
        
        # OpenCV circle with improved metrics
        ax2.imshow(opencv_circle, cmap='gray')
        ax2.set_title(f"OpenCV Circle (Improved)\nRoundness: {opencv_improved['roundness']:.4f}")
        ax2.axis('off')
        
        # Mathematical circle with original metrics
        ax3.imshow(math_circle, cmap='gray')
        ax3.set_title(f"Mathematical Circle (Original)\nRoundness: {math_original['roundness']:.4f}")
        ax3.axis('off')
        
        # Mathematical circle with improved metrics
        ax4.imshow(math_circle, cmap='gray')
        ax4.set_title(f"Mathematical Circle (Improved)\nRoundness: {math_improved['roundness']:.4f}")
        ax4.axis('off')
        
        plt.tight_layout()
        plt.savefig(output_dir / f"circle_radius_{radius}_comparison.png", dpi=150)
        plt.close()
    
    # Create comparison bar chart
    create_comparison_chart(results)
    
    # Test ellipses with different axis ratios
    test_ellipses()
    
    print("\nImproved metrics testing complete.")

def create_comparison_chart(results):
    """Create a bar chart comparing roundness values from different methods."""
    radii = [r['radius'] for r in results]
    
    # Set up bar positions
    bar_width = 0.2
    r1 = np.arange(len(radii))
    r2 = [x + bar_width for x in r1]
    r3 = [x + bar_width for x in r2]
    r4 = [x + bar_width for x in r3]
    
    # Create figure
    plt.figure(figsize=(12, 8))
    
    # Create bars
    plt.bar(r1, [r['opencv_original_roundness'] for r in results], 
            width=bar_width, label='OpenCV Circle (Original Metrics)', alpha=0.7)
    plt.bar(r2, [r['opencv_improved_roundness'] for r in results], 
            width=bar_width, label='OpenCV Circle (Improved Metrics)', alpha=0.7)
    plt.bar(r3, [r['math_original_roundness'] for r in results], 
            width=bar_width, label='Mathematical Circle (Original Metrics)', alpha=0.7)
    plt.bar(r4, [r['math_improved_roundness'] for r in results], 
            width=bar_width, label='Mathematical Circle (Improved Metrics)', alpha=0.7)
    
    # Add reference line for perfect roundness
    plt.axhline(y=1.0, color='r', linestyle='--', alpha=0.5, label='Perfect Circle (1.0)')
    
    # Customize chart
    plt.xlabel('Circle Radius')
    plt.ylabel('Roundness')
    plt.title('Comparison of Roundness Calculations')
    plt.xticks([r + bar_width*1.5 for r in range(len(radii))], radii)
    plt.ylim(0.8, 1.05)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / "roundness_comparison_chart.png", dpi=150)
    plt.close()

def test_ellipses():
    """Test improved metrics with ellipses of different axis ratios."""
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
        
        # Calculate metrics using both original and improved methods
        original = original_metrics(ellipse_mask)
        improved = improved_metrics(ellipse_mask)
        
        # Print results
        print(f"  Original Metrics:")
        print(f"    Ellipticity: {original['ellipticity']:.4f} (expected: {ratio:.1f})")
        print(f"    Roundness: {original['roundness']:.4f}")
        
        print(f"  Improved Metrics:")
        print(f"    Ellipticity: {improved['ellipticity']:.4f} (expected: {ratio:.1f})")
        print(f"    Roundness: {improved['roundness']:.4f}")
        
        # Store results
        ellipse_results.append({
            'ratio': ratio,
            'orig_ellipticity': original['ellipticity'],
            'impr_ellipticity': improved['ellipticity'],
            'orig_roundness': original['roundness'],
            'impr_roundness': improved['roundness']
        })
        
        # Create visualization
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
        
        ax1.imshow(ellipse_mask, cmap='gray')
        ax1.set_title(f"Ellipse {ratio}:1 - Original Metrics\nEllipticity: {original['ellipticity']:.2f}, Roundness: {original['roundness']:.2f}")
        ax1.axis('off')
        
        ax2.imshow(ellipse_mask, cmap='gray')
        ax2.set_title(f"Ellipse {ratio}:1 - Improved Metrics\nEllipticity: {improved['ellipticity']:.2f}, Roundness: {improved['roundness']:.2f}")
        ax2.axis('off')
        
        plt.tight_layout()
        plt.savefig(output_dir / f"ellipse_ratio_{ratio}_comparison.png", dpi=150)
        plt.close()
    
    # Create comparison charts for ellipses
    create_ellipse_comparison_chart(ellipse_results)

def create_ellipse_comparison_chart(results):
    """Create charts comparing ellipticity and roundness for ellipses."""
    ratios = [r['ratio'] for r in results]
    
    # Set up figures
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Plot ellipticity chart
    ax1.plot(ratios, ratios, 'r--', alpha=0.7, label='Expected')
    ax1.plot(ratios, [r['orig_ellipticity'] for r in results], 'bo-', alpha=0.7, label='Original Metrics')
    ax1.plot(ratios, [r['impr_ellipticity'] for r in results], 'go-', alpha=0.7, label='Improved Metrics')
    
    ax1.set_xlabel('Expected Axis Ratio')
    ax1.set_ylabel('Measured Ellipticity')
    ax1.set_title('Ellipticity Measurements')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # Plot roundness chart
    ax2.plot(ratios, [1/r if r > 0 else 0 for r in ratios], 'r--', alpha=0.7, label='Theoretical')
    ax2.plot(ratios, [r['orig_roundness'] for r in results], 'bo-', alpha=0.7, label='Original Metrics')
    ax2.plot(ratios, [r['impr_roundness'] for r in results], 'go-', alpha=0.7, label='Improved Metrics')
    
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
