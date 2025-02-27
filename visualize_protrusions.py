"""
Script to generate detailed visualizations of protrusion analysis.

This script creates shapes with various protrusion patterns and
generates visualizations showing how the new isolation approach works.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import cv2
from pathlib import Path

from seg_ana.core.synthetic import (
    create_shape_with_protrusions,
    create_mathematical_circle,
    create_circle_mask
)
from seg_ana.core.protrusion_analysis import (
    isolate_protrusions,
    analyze_all_protrusions,
    analyze_protrusion,
    summarize_protrusions
)

# Create output directory
output_dir = Path("./protrusion_visualization")
output_dir.mkdir(exist_ok=True)


def visualize_isolation_process(shape_name, mask):
    """Create a step-by-step visualization of the protrusion isolation process."""
    print(f"Visualizing isolation process for {shape_name}...")
    
    # Create a directory for this shape
    shape_dir = output_dir / shape_name
    shape_dir.mkdir(exist_ok=True)
    
    # 1. Original mask
    plt.figure(figsize=(8, 8))
    plt.imshow(mask, cmap='gray')
    plt.title(f"Original Mask: {shape_name}")
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(shape_dir / "1_original_mask.png", dpi=150)
    plt.close()
    
    # 2. Erosion to get the body
    # Use approximate radius for erosion kernel
    contours, _ = cv2.findContours(
        mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    original_contour = max(contours, key=cv2.contourArea)
    original_area = cv2.contourArea(original_contour)
    
    radius = int(np.sqrt(original_area / np.pi) * 0.4)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2*radius+1, 2*radius+1))
    
    body = cv2.erode(mask.astype(np.uint8), kernel)
    
    plt.figure(figsize=(8, 8))
    plt.imshow(body, cmap='gray')
    plt.title(f"Body After Erosion\nKernel radius: {radius} pixels")
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(shape_dir / "2_eroded_body.png", dpi=150)
    plt.close()
    
    # 3. Dilated body
    body_dilated = cv2.dilate(body, kernel)
    
    plt.figure(figsize=(8, 8))
    plt.imshow(body_dilated, cmap='gray')
    plt.title("Body After Dilation")
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(shape_dir / "3_dilated_body.png", dpi=150)
    plt.close()
    
    # 4. Protrusions (original - dilated body)
    protrusions_mask = cv2.subtract(mask.astype(np.uint8), body_dilated)
    
    plt.figure(figsize=(8, 8))
    plt.imshow(protrusions_mask, cmap='gray')
    plt.title("Isolated Protrusions\n(Original - Dilated Body)")
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(shape_dir / "4_isolated_protrusions.png", dpi=150)
    plt.close()
    
    # 5. Connected components labeling
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
        protrusions_mask, connectivity=8
    )
    
    # Create a colorful visualization of connected components
    label_viz = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)
    
    # Random colors for each label
    colors = np.random.randint(0, 255, size=(num_labels, 3), dtype=np.uint8)
    colors[0] = [0, 0, 0]  # Background is black
    
    for i in range(1, num_labels):
        label_viz[labels == i] = colors[i]
    
    plt.figure(figsize=(8, 8))
    plt.imshow(label_viz)
    plt.title(f"Connected Components\n{num_labels-1} potential protrusions")
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(shape_dir / "5_connected_components.png", dpi=150)
    plt.close()
    
    # 6. Final segmentation with size filtering
    body_dilated, protrusion_masks, visualization = isolate_protrusions(mask)
    
    plt.figure(figsize=(8, 8))
    plt.imshow(visualization)
    plt.title(f"Final Segmentation\n{len(protrusion_masks)} protrusions after filtering")
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(shape_dir / "6_final_segmentation.png", dpi=150)
    plt.close()
    
    # 7. Create a summary visualization
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    
    # Original
    axes[0].imshow(mask, cmap='gray')
    axes[0].set_title("1. Original Shape")
    axes[0].axis('off')
    
    # Eroded body
    axes[1].imshow(body, cmap='gray')
    axes[1].set_title("2. Eroded Body")
    axes[1].axis('off')
    
    # Dilated body
    axes[2].imshow(body_dilated, cmap='gray')
    axes[2].set_title("3. Dilated Body")
    axes[2].axis('off')
    
    # Isolated protrusions
    axes[3].imshow(protrusions_mask, cmap='gray')
    axes[3].set_title("4. Isolated Protrusions")
    axes[3].axis('off')
    
    # Connected components
    axes[4].imshow(label_viz)
    axes[4].set_title("5. Connected Components")
    axes[4].axis('off')
    
    # Final segmentation
    axes[5].imshow(visualization)
    axes[5].set_title("6. Final Segmentation")
    axes[5].axis('off')
    
    plt.tight_layout()
    plt.savefig(shape_dir / "isolation_process_summary.png", dpi=150)
    plt.close()
    
    # Now analyze protrusions and create detailed visualizations
    protrusion_results = analyze_all_protrusions(
        mask, visualize=True, output_dir=shape_dir
    )
    
    return protrusion_results


def create_variant_shapes():
    """Create different shapes to demonstrate protrusion analysis."""
    shapes = {}
    
    # 1. Perfect circle (no protrusions)
    shapes["perfect_circle"] = create_mathematical_circle(
        size=(512, 512), radius=100
    )
    
    # 2. Noisy circle
    shapes["noisy_circle"] = create_circle_mask(
        size=(512, 512), radius=100, noise=0.2
    )
    
    # 3. Regular protrusions
    shapes["regular_6_protrusions"] = create_shape_with_protrusions(
        size=(512, 512), radius=100, num_protrusions=6, 
        protrusion_size=25, protrusion_distance=1.3
    )
    
    # 4. Irregular protrusions (different sizes)
    base_shape = create_shape_with_protrusions(
        size=(512, 512), radius=100, num_protrusions=4,
        protrusion_size=10, protrusion_distance=1.3
    )
    
    # Add two larger protrusions manually
    center = (256, 256)
    angles = [45, 225]  # Degrees
    
    for angle in angles:
        angle_rad = np.radians(angle)
        protr_x = int(center[0] + 130 * np.cos(angle_rad))
        protr_y = int(center[1] + 130 * np.sin(angle_rad))
        
        # Draw a larger protrusion
        cv2.circle(
            base_shape, (protr_x, protr_y), 35, 1, -1
        )
        
        # Draw connecting arm
        arm_start_x = int(center[0] + 100 * np.cos(angle_rad))
        arm_start_y = int(center[1] + 100 * np.sin(angle_rad))
        
        cv2.line(
            base_shape, (arm_start_x, arm_start_y), (protr_x, protr_y), 1, 20
        )
    
    shapes["irregular_protrusions"] = base_shape
    
    # 5. Closely spaced protrusions
    shapes["closely_spaced"] = create_shape_with_protrusions(
        size=(512, 512), radius=100, num_protrusions=12,
        protrusion_size=15, protrusion_distance=1.2
    )
    
    return shapes


def create_comparison_visualization(shapes_results):
    """Create comparison visualizations of all shapes and their analyses."""
    print("Creating comparison visualizations...")
    
    # Create a grid showing all original shapes
    rows = int(np.ceil(len(shapes_results) / 3))
    fig, axes = plt.subplots(rows, 3, figsize=(15, 5 * rows))
    
    if rows == 1:
        axes = [axes]
    
    for i, (shape_name, results) in enumerate(shapes_results.items()):
        row, col = i // 3, i % 3
        if rows > 1:
            ax = axes[row][col]
        else:
            ax = axes[col]
        
        shape_dir = output_dir / shape_name
        
        # Load the final segmentation image
        segmentation_img = cv2.imread(str(shape_dir / "protrusion_segmentation.png"))
        segmentation_img = cv2.cvtColor(segmentation_img, cv2.COLOR_BGR2RGB)
        
        # Display the image
        ax.imshow(segmentation_img)
        
        # Set title with protrusion count
        ax.set_title(f"{shape_name.replace('_', ' ').title()}\n{results['protrusion_count']} protrusions")
        ax.axis('off')
    
    # Remove any empty subplots
    for i in range(len(shapes_results), rows * 3):
        row, col = i // 3, i % 3
        if rows > 1:
            fig.delaxes(axes[row][col])
    
    plt.tight_layout()
    plt.savefig(output_dir / "shape_comparison.png", dpi=150)
    plt.close()
    
    # Create a table of metrics
    metrics_data = []
    
    for shape_name, results in shapes_results.items():
        summary = summarize_protrusions(results)
        
        metrics_data.append({
            'Shape': shape_name.replace('_', ' ').title(),
            'Protrusion Count': summary['protrusion_count'],
            'Mean Length': f"{summary['mean_length']:.1f}",
            'Mean Width': f"{summary['mean_width']:.1f}",
            'Length CV': f"{summary['length_cv']:.3f}",
            'Spacing Uniformity': f"{summary['spacing_uniformity']:.3f}"
        })
    
    # Create a figure for the table
    fig, ax = plt.subplots(figsize=(12, len(metrics_data) + 2))
    ax.axis('tight')
    ax.axis('off')
    
    table = ax.table(
        cellText=[[d[k] for k in d.keys()] for d in metrics_data],
        colLabels=list(metrics_data[0].keys()),
        loc='center',
        cellLoc='center'
    )
    
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.5)
    
    plt.title("Protrusion Analysis Metrics Comparison", fontsize=16, pad=20)
    plt.tight_layout()
    plt.savefig(output_dir / "metrics_comparison_table.png", dpi=150)
    plt.close()


def main():
    """Run the protrusion visualization and analysis."""
    print(f"Generating protrusion analysis visualizations")
    print(f"Results will be saved to: {output_dir.absolute()}")
    
    # Create various shapes
    shapes = create_variant_shapes()
    
    # Analyze each shape and create visualizations
    results = {}
    
    for shape_name, mask in shapes.items():
        print(f"\nAnalyzing {shape_name}...")
        
        # Save the original shape
        np.save(output_dir / f"{shape_name}.npy", mask)
        
        # Visualize isolation process and analyze protrusions
        protrusion_results = visualize_isolation_process(shape_name, mask)
        results[shape_name] = protrusion_results
    
    # Create comparison visualizations
    create_comparison_visualization(results)
    
    print(f"\nAll visualizations complete. Results saved to: {output_dir.absolute()}")


if __name__ == "__main__":
    main()
