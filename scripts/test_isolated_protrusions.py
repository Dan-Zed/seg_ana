"""
Test script to demonstrate the enhanced protrusion isolation and analysis.

This script creates shapes with known numbers of protrusions and
uses the new approach to isolate and analyze each protrusion individually.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import cv2
from pathlib import Path
import pandas as pd

from seg_ana.core.synthetic import (
    create_shape_with_protrusions,
    create_mathematical_circle
)
from seg_ana.core.metrics import calculate_all_metrics
from seg_ana.core.protrusion_analysis import (
    isolate_protrusions,
    analyze_all_protrusions,
    summarize_protrusions
)

# Create output directory
output_dir = Path("./isolated_protrusions_test")
output_dir.mkdir(exist_ok=True)


def compare_methods(mask, expected_count, filename):
    """Compare standard metrics with enhanced protrusion analysis."""
    # Standard metrics
    metrics = calculate_all_metrics(mask)
    
    # Enhanced protrusion analysis
    protrusion_dir = output_dir / filename.replace(".png", "")
    protrusion_dir.mkdir(exist_ok=True)
    
    results = analyze_all_protrusions(
        mask, visualize=True, output_dir=protrusion_dir
    )
    
    summary = summarize_protrusions(results)
    
    # Create comparison image
    fig, axes = plt.subplots(1, 2, figsize=(15, 7))
    
    # Original mask with standard metrics
    axes[0].imshow(mask, cmap='gray')
    axes[0].set_title(f"Standard Metrics\nDetected: {metrics['protrusions']} protrusions")
    
    # Add text with standard metrics
    metrics_text = (
        f"Protrusions: {metrics['protrusions']}\n"
        f"Roundness: {metrics['roundness']:.4f}\n"
        f"Solidity: {metrics['solidity']:.4f}\n"
        f"Ellipticity: {metrics['ellipticity']:.4f}"
    )
    axes[0].text(
        0.05, 0.95, metrics_text,
        transform=axes[0].transAxes,
        fontsize=10,
        verticalalignment='top',
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    )
    
    # Load the segmentation image from the enhanced method
    segmentation_img = cv2.imread(str(protrusion_dir / "protrusion_segmentation.png"))
    segmentation_img = cv2.cvtColor(segmentation_img, cv2.COLOR_BGR2RGB)
    
    # Display enhanced analysis
    axes[1].imshow(segmentation_img)
    axes[1].set_title(f"Enhanced Analysis\nDetected: {summary['protrusion_count']} protrusions")
    
    # Add text with enhanced metrics
    enhanced_text = (
        f"Protrusions: {summary['protrusion_count']}\n"
        f"Mean length: {summary['mean_length']:.1f} pixels\n"
        f"Mean width: {summary['mean_width']:.1f} pixels\n"
        f"Spacing uniformity: {summary['spacing_uniformity']:.3f}"
    )
    axes[1].text(
        0.05, 0.95, enhanced_text,
        transform=axes[1].transAxes,
        fontsize=10,
        verticalalignment='top',
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    )
    
    # Add expected count as caption
    fig.suptitle(f"Shape with {expected_count} expected protrusions", fontsize=16)
    
    # Save comparison
    plt.tight_layout()
    plt.savefig(output_dir / filename, dpi=150)
    plt.close()
    
    return {
        'expected': expected_count,
        'standard_count': metrics['protrusions'],
        'enhanced_count': summary['protrusion_count'],
        'standard_roundness': metrics['roundness'],
        'standard_solidity': metrics['solidity'],
        'mean_protrusion_length': summary['mean_length'],
        'mean_protrusion_width': summary['mean_width'],
        'spacing_uniformity': summary['spacing_uniformity']
    }


def main():
    """Run the test and generate visualizations."""
    print(f"Testing enhanced protrusion isolation and analysis")
    print(f"Results will be saved to: {output_dir.absolute()}")
    
    # Test various numbers of protrusions
    protrusion_counts = [0, 3, 6, 9]
    results = []
    
    for count in protrusion_counts:
        print(f"\nTesting with {count} protrusions...")
        
        # Create shape with known number of protrusions
        if count == 0:
            # Perfect circle has no protrusions
            shape = create_mathematical_circle(size=(512, 512), radius=100)
        else:
            shape = create_shape_with_protrusions(
                size=(512, 512),
                radius=100,
                num_protrusions=count,
                protrusion_size=25,
                protrusion_distance=1.3
            )
        
        # Save the original shape
        np.save(output_dir / f"shape_with_{count}_protrusions.npy", shape)
        
        # Compare standard and enhanced methods
        result = compare_methods(
            shape, count, f"comparison_{count}_protrusions.png"
        )
        
        results.append(result)
    
    # Save results table
    df = pd.DataFrame(results)
    print("\nResults summary:")
    print(df)
    
    # Save to CSV
    df.to_csv(output_dir / "protrusion_comparison_results.csv", index=False)
    
    # Create a bar chart comparing methods
    plt.figure(figsize=(12, 6))
    
    x = np.arange(len(protrusion_counts))
    width = 0.35
    
    plt.bar(x - width/2, df['standard_count'], width, label='Standard Method')
    plt.bar(x + width/2, df['enhanced_count'], width, label='Enhanced Method')
    
    plt.xlabel('Expected Protrusion Count')
    plt.ylabel('Detected Protrusion Count')
    plt.title('Comparison of Protrusion Detection Methods')
    plt.xticks(x, protrusion_counts)
    plt.legend()
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    plt.savefig(output_dir / "method_comparison_chart.png", dpi=150)
    plt.close()
    
    # Now try with a noisy circle
    print("\nTesting with a noisy circle...")
    from seg_ana.core.synthetic import create_circle_mask
    
    noisy_circle = create_circle_mask(
        size=(512, 512),
        radius=100,
        noise=0.2
    )
    
    # Compare methods
    noisy_result = compare_methods(
        noisy_circle, 0, "comparison_noisy_circle.png"
    )
    
    print(f"\nNoisy circle results:")
    print(f"  Standard method: {noisy_result['standard_count']} protrusions")
    print(f"  Enhanced method: {noisy_result['enhanced_count']} protrusions")
    
    print(f"\nAll tests complete. Results saved to: {output_dir.absolute()}")


if __name__ == "__main__":
    main()
