"""
Script to generate visualizations of how each metric is calculated.

This creates a set of images illustrating each metric calculation with
visual explanations of how the measurements are derived.
"""
import os
import numpy as np
import matplotlib.pyplot as plt
import cv2
from pathlib import Path
import math

from seg_ana.core.synthetic import (
    create_circle_mask,
    create_ellipse_mask,
    create_shape_with_protrusions,
    create_mathematical_circle
)
from seg_ana.core.metrics_improved import (
    calculate_all_metrics,
    get_largest_contour
)

# Create output directory
output_dir = Path("./metrics_visualization")
output_dir.mkdir(exist_ok=True)


def create_figure_with_text(title, size=(10, 8)):
    """Create a figure with title."""
    fig, ax = plt.subplots(figsize=size)
    ax.set_title(title, fontsize=14)
    return fig, ax


def add_text_block(ax, text, position=(0.05, 0.95), va='top'):
    """Add a text block to an axis."""
    ax.text(
        position[0], position[1], text,
        transform=ax.transAxes,
        fontsize=12,
        verticalalignment=va,
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.4)
    )


def visualize_area_calculation():
    """Visualize how area is calculated."""
    print("Generating area visualization...")
    
    # Create a circle
    mask = create_mathematical_circle(size=(512, 512), radius=100)
    contour = get_largest_contour(mask)
    area = cv2.contourArea(contour)
    
    # Create figure
    fig, ax = create_figure_with_text("Area Calculation", size=(8, 8))
    
    # Display mask
    ax.imshow(mask, cmap='gray')
    
    # Overlay contour
    contour_points = contour.squeeze()
    ax.plot(contour_points[:, 0], contour_points[:, 1], 'r-', linewidth=2)
    
    # No text explanation as per request
    
    # Draw a filled semi-transparent overlay
    filled_mask = np.zeros((512, 512, 4))
    filled_mask[mask > 0] = [1, 0, 0, 0.3]  # Semi-transparent red
    ax.imshow(filled_mask)
    
    # Save the figure
    plt.tight_layout()
    plt.savefig(output_dir / "area_calculation.png", dpi=150)
    plt.close()


def visualize_perimeter_calculation():
    """Visualize how perimeter is calculated."""
    print("Generating perimeter visualization...")
    
    # Create a circle
    mask = create_mathematical_circle(size=(512, 512), radius=100)
    contour = get_largest_contour(mask)
    
    # Get perimeter with and without smoothing
    perimeter_raw = cv2.arcLength(contour, True)
    
    # Smooth contour
    epsilon = 0.01 * perimeter_raw
    smooth_contour = cv2.approxPolyDP(contour, epsilon, True)
    perimeter_smooth = cv2.arcLength(smooth_contour, True)
    
    # Create figure
    fig, ax = create_figure_with_text("Perimeter Calculation", size=(8, 8))
    
    # Display mask
    ax.imshow(mask, cmap='gray')
    
    # Draw original contour
    contour_points = contour.squeeze()
    ax.plot(contour_points[:, 0], contour_points[:, 1], 'r-', linewidth=2, alpha=0.5, label='Original contour')
    
    # Draw smoothed contour
    smooth_points = smooth_contour.squeeze()
    ax.plot(smooth_points[:, 0], smooth_points[:, 1], 'b-', linewidth=2, label='Smoothed contour')
    
    # No text explanation as per request
    
    # Add legend
    ax.legend(loc='lower right')
    
    # Save the figure
    plt.tight_layout()
    plt.savefig(output_dir / "perimeter_calculation.png", dpi=150)
    plt.close()


def visualize_roundness_calculation():
    """Visualize how roundness is calculated."""
    print("Generating roundness visualization...")
    
    # Create a perfect circle and an ellipse for comparison
    circle_mask = create_mathematical_circle(size=(512, 512), radius=100)
    ellipse_mask = create_ellipse_mask(size=(512, 512), axes=(150, 75))
    
    # Calculate metrics
    circle_metrics = calculate_all_metrics(circle_mask)
    ellipse_metrics = calculate_all_metrics(ellipse_mask)
    
    # Create a two-panel figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 7))
    fig.suptitle("Roundness Calculation", fontsize=16)
    
    # Display circle
    ax1.imshow(circle_mask, cmap='gray')
    ax1.set_title(f"Perfect Circle: Roundness = {circle_metrics['roundness']:.4f}")
    
    # Display ellipse
    ax2.imshow(ellipse_mask, cmap='gray')
    ax2.set_title(f"Ellipse (2:1 ratio): Roundness = {ellipse_metrics['roundness']:.4f}")
    
    # Add overall explanation
    explanation = (
        "Roundness = 4π × Area / Perimeter²\n\n"
        "This ratio is maximized (=1.0) for a perfect circle.\n"
        "For other shapes, roundness < 1.0.\n\n"
        "Alternative calculation method:\n"
        "Roundness = Perimeter of equivalent circle / Actual perimeter\n\n"
        "We choose the better of these two calculations."
    )
    
    # No text box explanation as per request
    
    # Save the figure
    plt.tight_layout()
    plt.savefig(output_dir / "roundness_calculation.png", dpi=150)
    plt.close()


def visualize_ellipticity_calculation():
    """Visualize how ellipticity is calculated."""
    print("Generating ellipticity visualization...")
    
    # Create ellipses with different aspect ratios - always horizontal
    # Using angle=0 ensures the major axis is horizontal and minor axis is vertical
    ellipse1_mask = create_ellipse_mask(size=(512, 512), axes=(100, 100), angle=0)  # Circle (1:1)
    ellipse2_mask = create_ellipse_mask(size=(512, 512), axes=(150, 75), angle=0)   # 2:1 ratio
    ellipse3_mask = create_ellipse_mask(size=(512, 512), axes=(180, 60), angle=0)   # 3:1 ratio
    
    # Calculate metrics
    metrics1 = calculate_all_metrics(ellipse1_mask)
    metrics2 = calculate_all_metrics(ellipse2_mask)
    metrics3 = calculate_all_metrics(ellipse3_mask)
    
    # Create a figure with three panels
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle("Ellipticity Calculation", fontsize=16)
    
    # Display ellipses
    ax1.imshow(ellipse1_mask, cmap='gray')
    ax1.set_title(f"1:1 Ratio\nEllipticity = {metrics1['ellipticity']:.2f}")
    
    ax2.imshow(ellipse2_mask, cmap='gray')
    ax2.set_title(f"2:1 Ratio\nEllipticity = {metrics2['ellipticity']:.2f}")
    
    ax3.imshow(ellipse3_mask, cmap='gray')
    ax3.set_title(f"3:1 Ratio\nEllipticity = {metrics3['ellipticity']:.2f}")
    
    # Add overlays to show major and minor axes
    def plot_axes(ax, mask, metrics):
        # For our visualization, we want horizontal major axis and vertical minor axis
        # regardless of how OpenCV fits the ellipse
        
        center = (mask.shape[1] // 2, mask.shape[0] // 2)
        major = metrics['major_axis'] / 2
        minor = metrics['minor_axis'] / 2
        
        # Major axis (horizontal)
        x1 = center[0] - major
        y1 = center[1]
        x2 = center[0] + major
        y2 = center[1]
        
        # Minor axis (vertical)
        x3 = center[0]
        y3 = center[1] - minor
        x4 = center[0]
        y4 = center[1] + minor
        
        # Plot horizontal major axis in red
        ax.plot([x1, x2], [y1, y2], 'r-', linewidth=2, label='Major axis')
        # Plot vertical minor axis in green
        ax.plot([x3, x4], [y3, y4], 'g-', linewidth=2, label='Minor axis')
        
        # Add length labels
        ax.text((x1 + x2) / 2, (y1 + y2) / 2, f"{metrics['major_axis']:.1f}", 
                color='red', fontsize=10, ha='center', va='center', 
                bbox=dict(facecolor='white', alpha=0.7))
        ax.text((x3 + x4) / 2, (y3 + y4) / 2, f"{metrics['minor_axis']:.1f}", 
                color='green', fontsize=10, ha='center', va='center',
                bbox=dict(facecolor='white', alpha=0.7))
    
    plot_axes(ax1, ellipse1_mask, metrics1)
    plot_axes(ax2, ellipse2_mask, metrics2)
    plot_axes(ax3, ellipse3_mask, metrics3)
    
    # Add explanation
    explanation = (
        "Ellipticity = Major axis / Minor axis\n\n"
        "A perfect circle has ellipticity = 1.0\n"
        "More elongated shapes have higher ellipticity values.\n\n"
        "The axes are found by fitting an ellipse to the contour\n"
        "using OpenCV's fitEllipse function."
    )
    
    # No text box explanation as per request
    
    # Add a single legend for the entire figure
    handles, labels = ax1.get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper right', bbox_to_anchor=(0.95, 0.95))
    
    # Save the figure
    plt.tight_layout()
    plt.savefig(output_dir / "ellipticity_calculation.png", dpi=150)
    plt.close()


def visualize_solidity_calculation():
    """Visualize how solidity is calculated."""
    print("Generating solidity visualization...")
    
    # Create shapes with different solidity values
    circle_mask = create_mathematical_circle(size=(512, 512), radius=100)
    shape1 = create_shape_with_protrusions(
        size=(512, 512), radius=100, num_protrusions=0, protrusion_size=0
    )
    shape2 = create_shape_with_protrusions(
        size=(512, 512), radius=100, num_protrusions=3, protrusion_size=20
    )
    shape3 = create_shape_with_protrusions(
        size=(512, 512), radius=100, num_protrusions=6, protrusion_size=30
    )
    
    # Calculate metrics
    metrics1 = calculate_all_metrics(shape1)
    metrics2 = calculate_all_metrics(shape2)
    metrics3 = calculate_all_metrics(shape3)
    
    # Create a figure with three panels
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle("Solidity Calculation", fontsize=16)
    
    # Display shapes
    contour1 = get_largest_contour(shape1)
    hull1 = cv2.convexHull(contour1)
    
    contour2 = get_largest_contour(shape2)
    hull2 = cv2.convexHull(contour2)
    
    contour3 = get_largest_contour(shape3)
    hull3 = cv2.convexHull(contour3)
    
    def plot_solidity(ax, mask, contour, hull, metrics):
        # Display mask
        ax.imshow(mask, cmap='gray')
        
        # Draw contour and hull
        contour_points = contour.squeeze()
        hull_points = hull.squeeze()
        
        ax.plot(contour_points[:, 0], contour_points[:, 1], 'r-', linewidth=2, label='Contour')
        ax.plot(hull_points[:, 0], hull_points[:, 1], 'g-', linewidth=2, label='Convex hull')
        
        # Set title with metrics
        ax.set_title(f"Solidity = {metrics['solidity']:.4f}")
        
        # Highlight convex deficiency areas
        hull_mask = np.zeros_like(mask)
        cv2.drawContours(hull_mask, [hull], 0, 1, -1)
        
        # Create a mask for areas in hull but not in original contour
        deficiency = np.logical_and(hull_mask > 0, mask == 0)
        
        # Overlay these areas in semi-transparent blue
        deficiency_overlay = np.zeros((mask.shape[0], mask.shape[1], 4))
        deficiency_overlay[deficiency] = [0, 0, 1, 0.5]  # Semi-transparent blue
        ax.imshow(deficiency_overlay)
    
    plot_solidity(ax1, shape1, contour1, hull1, metrics1)
    plot_solidity(ax2, shape2, contour2, hull2, metrics2)
    plot_solidity(ax3, shape3, contour3, hull3, metrics3)
    
    # Add explanation
    explanation = (
        "Solidity = Contour Area / Convex Hull Area\n\n"
        "The blue highlighted areas show the 'convex deficiency' (regions inside the hull but outside the contour).\n\n"
        "A perfect convex shape has solidity = 1.0\n"
        "Shapes with concavities or indentations have solidity < 1.0.\n"
        "The more concave a shape, the lower its solidity value."
    )
    
    # No text box explanation as per request
    
    # Add a single legend for the entire figure
    handles, labels = ax1.get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper right', bbox_to_anchor=(0.95, 0.95))
    
    # Save the figure
    plt.tight_layout()
    plt.savefig(output_dir / "solidity_calculation.png", dpi=150)
    plt.close()


def visualize_protrusion_calculation():
    """Visualize how protrusions are calculated."""
    print("Generating protrusion visualization...")
    
    # Create shapes with different numbers of protrusions
    shape1 = create_mathematical_circle(size=(512, 512), radius=100)  # 0 protrusions
    shape2 = create_shape_with_protrusions(
        size=(512, 512), radius=100, num_protrusions=3, protrusion_size=25
    )
    shape3 = create_shape_with_protrusions(
        size=(512, 512), radius=100, num_protrusions=6, protrusion_size=25
    )
    
    # Calculate metrics
    metrics1 = calculate_all_metrics(shape1)
    metrics2 = calculate_all_metrics(shape2)
    metrics3 = calculate_all_metrics(shape3)
    
    # Create a figure
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle("Protrusion Detection", fontsize=16)
    
    def visualize_protrusion_distances(ax, mask, title):
        # Get contour and hull
        contour = get_largest_contour(mask)
        hull = cv2.convexHull(contour)
        
        # Calculate distances from contour to hull
        contour_points = contour.squeeze()
        hull_points = hull.squeeze()
        
        distances = np.min(
            np.linalg.norm(contour_points[:, None] - hull_points[None], axis=2),
            axis=1
        )
        
        # Find points that exceed threshold
        threshold = 5.0
        protrusion_points = distances > threshold
        
        # Display mask
        ax.imshow(mask, cmap='gray')
        
        # Draw hull
        ax.plot(hull_points[:, 0], hull_points[:, 1], 'g-', linewidth=2, label='Convex hull')
        
        # Color contour points based on distance
        # Regular contour points in blue
        regular_indices = np.where(~protrusion_points)[0]
        ax.scatter(
            contour_points[regular_indices, 0],
            contour_points[regular_indices, 1],
            c='blue', s=10, alpha=0.5, label='Regular points'
        )
        
        # Protrusion points in red
        protrusion_indices = np.where(protrusion_points)[0]
        if len(protrusion_indices) > 0:
            ax.scatter(
                contour_points[protrusion_indices, 0],
                contour_points[protrusion_indices, 1],
                c='red', s=20, label='Protrusion points'
            )
        
        ax.set_title(title)
    
    visualize_protrusion_distances(
        ax1, shape1, f"Circle: {metrics1['protrusions']} protrusions detected"
    )
    visualize_protrusion_distances(
        ax2, shape2, f"3 Protrusions: {metrics2['protrusions']} detected"
    )
    visualize_protrusion_distances(
        ax3, shape3, f"6 Protrusions: {metrics3['protrusions']} detected"
    )
    
    # No text box explanation as per request
    
    # Add a single legend for the entire figure
    handles, labels = ax3.get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper right', bbox_to_anchor=(0.95, 0.95))
    
    # Save the figure
    plt.tight_layout()
    plt.savefig(output_dir / "protrusion_calculation.png", dpi=150)
    plt.close()


def visualize_comparison_perfect_vs_digital():
    """Compare mathematical vs. OpenCV drawn circles."""
    print("Generating mathematical vs. OpenCV circle comparison...")
    
    # Create both types of circles
    math_circle = create_mathematical_circle(size=(512, 512), radius=100)
    opencv_circle = create_circle_mask(size=(512, 512), radius=100)
    
    # Calculate metrics
    math_metrics = calculate_all_metrics(math_circle)
    opencv_metrics = calculate_all_metrics(opencv_circle)
    
    # Create a figure with two panels
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
    fig.suptitle("Perfect vs. Digital Circle Comparison", fontsize=16)
    
    # Display circles
    ax1.imshow(math_circle, cmap='gray')
    ax1.set_title(f"Mathematical Circle\nRoundness: {math_metrics['roundness']:.4f}")
    
    ax2.imshow(opencv_circle, cmap='gray')
    ax2.set_title(f"OpenCV Circle\nRoundness: {opencv_metrics['roundness']:.4f}")
    
    # Add zoomed insets to show the pixel-level differences
    from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes, mark_inset
    
    # Add inset for mathematical circle
    axins1 = zoomed_inset_axes(ax1, zoom=6, loc='upper right')
    axins1.imshow(math_circle, cmap='gray', interpolation='none')
    axins1.set_xlim(245, 265)
    axins1.set_ylim(245, 265)
    axins1.set_xticks([])
    axins1.set_yticks([])
    mark_inset(ax1, axins1, loc1=1, loc2=3, fc="none", ec="red")
    
    # Add inset for OpenCV circle
    axins2 = zoomed_inset_axes(ax2, zoom=6, loc='upper right')
    axins2.imshow(opencv_circle, cmap='gray', interpolation='none')
    axins2.set_xlim(245, 265)
    axins2.set_ylim(245, 265)
    axins2.set_xticks([])
    axins2.set_yticks([])
    mark_inset(ax2, axins2, loc1=1, loc2=3, fc="none", ec="red")
    
    # Add explanation
    explanation = (
        "Two methods of creating a 'perfect' circle in digital images:\n\n"
        "1. Mathematical Circle: Uses exact Euclidean distance calculation\n"
        "   - Includes all pixels where distance to center ≤ radius\n"
        "   - Produces more accurate roundness values\n\n"
        "2. OpenCV Drawing: Uses a drawing algorithm\n"
        "   - May have small artifacts due to rasterization\n"
        "   - Faster but slightly less precise\n\n"
        "The zoomed insets show pixel-level differences in the boundary."
    )
    
    # No text box explanation as per request
    
    # Save the figure
    plt.tight_layout()
    plt.savefig(output_dir / "perfect_vs_digital_circle.png", dpi=150)
    plt.close()


def create_summary_page():
    """Create a summary page showing all metrics on a single shape."""
    print("Generating metrics summary visualization...")
    
    # Create a shape with interesting properties
    shape = create_shape_with_protrusions(
        size=(512, 512), radius=100, num_protrusions=4, protrusion_size=30
    )
    
    # Calculate metrics
    metrics = calculate_all_metrics(shape)
    
    # Get contour and hull
    contour = get_largest_contour(shape)
    hull = cv2.convexHull(contour)
    
    # Fit ellipse
    ellipse = cv2.fitEllipse(contour)
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 10))
    fig.suptitle("Comprehensive Shape Analysis", fontsize=16)
    
    # Display shape
    ax.imshow(shape, cmap='gray')
    
    # Draw contour
    contour_points = contour.squeeze()
    ax.plot(contour_points[:, 0], contour_points[:, 1], 'r-', linewidth=2, label='Contour')
    
    # Draw hull
    hull_points = hull.squeeze()
    ax.plot(hull_points[:, 0], hull_points[:, 1], 'g-', linewidth=2, label='Convex hull')
    
    # Draw fitted ellipse
    center, axes, angle = ellipse
    from matplotlib.patches import Ellipse
    e = Ellipse(
        xy=center, width=axes[0], height=axes[1],
        angle=angle, linewidth=2, fill=False, edgecolor='blue', label='Fitted ellipse'
    )
    ax.add_patch(e)
    
    # Add metrics text
    metrics_text = (
        f"Area: {metrics['area']:.1f} pixels²\n"
        f"Perimeter: {metrics['perimeter']:.1f} pixels\n"
        f"Roundness: {metrics['roundness']:.4f}\n"
        f"Ellipticity: {metrics['ellipticity']:.4f}\n"
        f"Solidity: {metrics['solidity']:.4f}\n"
        f"Protrusions: {metrics['protrusions']}"
    )
    
    # Add text box with metrics
    ax.text(0.02, 0.98, metrics_text, transform=ax.transAxes, fontsize=12,
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # Add legend
    ax.legend(loc='lower right')
    
    # Save the figure
    plt.tight_layout()
    plt.savefig(output_dir / "metrics_summary.png", dpi=150)
    plt.close()


def main():
    """Generate all visualizations."""
    print(f"Generating metric visualization images in: {output_dir.absolute()}")
    
    # Generate individual visualizations
    visualize_area_calculation()
    visualize_perimeter_calculation()
    visualize_roundness_calculation()
    visualize_ellipticity_calculation()
    visualize_solidity_calculation()
    visualize_protrusion_calculation()
    visualize_comparison_perfect_vs_digital()
    create_summary_page()
    
    # Create index visualization
    fig, axes = plt.subplots(3, 3, figsize=(15, 15))
    axes = axes.flatten()
    
    # List of visualization files
    viz_files = [
        "area_calculation.png",
        "perimeter_calculation.png",
        "roundness_calculation.png",
        "ellipticity_calculation.png",
        "solidity_calculation.png",
        "protrusion_calculation.png",
        "perfect_vs_digital_circle.png",
        "metrics_summary.png"
    ]
    
    # Display each visualization
    for i, file in enumerate(viz_files):
        if i >= len(axes):
            break
            
        # Load image
        img = plt.imread(output_dir / file)
        
        # Display in grid
        axes[i].imshow(img)
        axes[i].set_title(file.replace('_', ' ').replace('.png', '').title())
        axes[i].axis('off')
    
    # Remove empty subplots
    for i in range(len(viz_files), len(axes)):
        fig.delaxes(axes[i])
    
    # Set overall title
    fig.suptitle("Metrics Visualization Index", fontsize=20)
    
    # Save index
    plt.tight_layout()
    plt.savefig(output_dir / "index.png", dpi=150)
    plt.close()
    
    print(f"All visualizations complete. Open {output_dir / 'index.png'} for an overview.")


if __name__ == "__main__":
    main()
