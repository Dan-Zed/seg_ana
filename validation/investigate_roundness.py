"""
Script for investigating the roundness calculation in detail.
"""
import numpy as np
import cv2
import matplotlib.pyplot as plt
from pathlib import Path

from seg_ana.core.synthetic import create_circle_mask
from seg_ana.core.metrics import get_largest_contour, calculate_all_metrics

# Create output directory if it doesn't exist
output_dir = Path("./validation_output")
output_dir.mkdir(exist_ok=True)

def visualize_contour_points(mask, contour, save_path=None):
    """Visualize contour points on top of a mask."""
    fig, ax = plt.subplots(figsize=(10, 10))
    
    # Show mask
    ax.imshow(mask, cmap='gray')
    
    # Plot contour points
    contour_points = contour.squeeze()
    ax.scatter(contour_points[:, 0], contour_points[:, 1], c='r', s=20, alpha=0.7)
    
    # Draw contour line
    ax.plot(contour_points[:, 0], contour_points[:, 1], 'r-', linewidth=1.5)
    
    ax.set_title("Contour Points")
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    else:
        plt.show()

def print_contour_stats(contour, radius):
    """Print statistics about a contour relative to a perfect circle."""
    contour_points = contour.squeeze()
    
    # Calculate center of mass
    center_x = np.mean(contour_points[:, 0])
    center_y = np.mean(contour_points[:, 1])
    
    # Calculate distances from center
    distances = np.sqrt(
        (contour_points[:, 0] - center_x)**2 + 
        (contour_points[:, 1] - center_y)**2
    )
    
    print(f"Contour Statistics:")
    print(f"  Number of points:       {len(contour_points)}")
    print(f"  Center of mass:         ({center_x:.2f}, {center_y:.2f})")
    print(f"  Expected radius:        {radius:.2f}")
    print(f"  Average radius:         {np.mean(distances):.2f}")
    print(f"  Min radius:             {np.min(distances):.2f}")
    print(f"  Max radius:             {np.max(distances):.2f}")
    print(f"  Std dev of radius:      {np.std(distances):.2f}")
    print(f"  CV of radius (%):       {np.std(distances)/np.mean(distances)*100:.2f}%")

def calculate_analytical_roundness(contour):
    """Calculate roundness using the analytical formula."""
    area = cv2.contourArea(contour)
    perimeter = cv2.arcLength(contour, True)
    return 4 * np.pi * area / (perimeter**2)

def investigate_radius_dependence():
    """Investigate how roundness varies with circle radius."""
    print("\nInvestigating radius dependence of roundness calculation...")
    
    radii = list(range(10, 101, 10))  # 10, 20, ..., 100
    roundness_values = []
    
    for radius in radii:
        # Create circle
        mask = create_circle_mask(size=(256, 256), radius=radius, noise=0.0)
        
        # Get contour and calculate roundness
        contour = get_largest_contour(mask.astype(np.uint8))
        roundness = calculate_analytical_roundness(contour)
        roundness_values.append(roundness)
        
        print(f"Radius {radius:3d}: Roundness = {roundness:.4f}")
    
    # Plot results
    plt.figure(figsize=(10, 6))
    plt.plot(radii, roundness_values, 'o-', linewidth=2)
    plt.axhline(y=1.0, color='r', linestyle='--', alpha=0.7)
    plt.xlabel('Circle Radius (pixels)')
    plt.ylabel('Calculated Roundness')
    plt.title('Roundness vs. Circle Radius')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / 'roundness_vs_radius.png', dpi=150)
    plt.close()

def fix_check():
    """Check a potential fix for the roundness calculation."""
    print("\nChecking potential fixes for roundness calculation...")
    
    # Create a circle
    radius = 50
    mask = create_circle_mask(size=(256, 256), radius=radius, noise=0.0)
    
    # Get contour
    contour = get_largest_contour(mask.astype(np.uint8))
    
    # Calculate metrics
    area = cv2.contourArea(contour)
    perimeter = cv2.arcLength(contour, True)
    
    # Standard calculation
    standard_roundness = 4 * np.pi * area / (perimeter**2)
    
    # Alternative calculations to try
    # 1. Using approximated contour with different epsilon values
    roundness_values = [standard_roundness]
    epsilon_values = [0.001, 0.01, 0.02, 0.05, 0.1]
    
    for epsilon in epsilon_values:
        # Approximate contour
        epsilon_absolute = epsilon * cv2.arcLength(contour, True)
        approx_contour = cv2.approxPolyDP(contour, epsilon_absolute, True)
        
        # Calculate metrics
        approx_area = cv2.contourArea(approx_contour)
        approx_perimeter = cv2.arcLength(approx_contour, True)
        approx_roundness = 4 * np.pi * approx_area / (approx_perimeter**2)
        
        roundness_values.append(approx_roundness)
        
        print(f"Epsilon {epsilon:.3f}: Points = {len(approx_contour)}, Roundness = {approx_roundness:.4f}")
    
    # Plot results
    labels = ['Original'] + [f'ε = {e}' for e in epsilon_values]
    plt.figure(figsize=(10, 6))
    plt.bar(labels, roundness_values, alpha=0.7)
    plt.axhline(y=1.0, color='r', linestyle='--', alpha=0.7)
    plt.ylabel('Calculated Roundness')
    plt.title('Roundness with Different Contour Approximations')
    plt.ylim(0.8, 1.05)
    plt.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig(output_dir / 'roundness_approximations.png', dpi=150)
    plt.close()

def export_modified_mask(save_npy=True):
    """Create a modified version of the circle mask to fix roundness."""
    # Create a circle using a different approach
    size = (256, 256)
    radius = 50
    center = (size[1] // 2, size[0] // 2)
    
    # Method 1: Use OpenCV circle drawing
    mask_cv = np.zeros(size, dtype=np.uint8)
    cv2.circle(mask_cv, center, radius, 1, -1)
    
    # Method 2: Use mathematical distance calculation
    y, x = np.ogrid[:size[0], :size[1]]
    dist_from_center = np.sqrt((x - center[0])**2 + (y - center[1])**2)
    mask_math = dist_from_center <= radius
    
    # Calculate metrics for both
    contour_cv = get_largest_contour(mask_cv)
    contour_math = get_largest_contour(mask_math.astype(np.uint8))
    
    roundness_cv = calculate_analytical_roundness(contour_cv)
    roundness_math = calculate_analytical_roundness(contour_math)
    
    print(f"\nComparison of mask creation methods:")
    print(f"  OpenCV circle:    Roundness = {roundness_cv:.4f}")
    print(f"  Mathematical:     Roundness = {roundness_math:.4f}")
    
    # Save both masks for comparison
    if save_npy:
        np.save(output_dir / "circle_mask_cv.npy", mask_cv)
        np.save(output_dir / "circle_mask_math.npy", mask_math)
    
    # Visualize both
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
    
    ax1.imshow(mask_cv, cmap='gray')
    ax1.set_title(f"OpenCV Circle\nRoundness: {roundness_cv:.4f}")
    
    ax2.imshow(mask_math, cmap='gray')
    ax2.set_title(f"Mathematical Circle\nRoundness: {roundness_math:.4f}")
    
    plt.tight_layout()
    plt.savefig(output_dir / 'circle_comparison.png', dpi=150)
    plt.close()
    
    return mask_math if roundness_math > roundness_cv else mask_cv

if __name__ == "__main__":
    print(f"Investigating roundness calculation...")
    print(f"Saving output to {output_dir.absolute()}")
    
    # Create a perfect circle
    radius = 50
    circle_mask = create_circle_mask(size=(256, 256), radius=radius, noise=0.0)
    
    # Get contour
    contour = get_largest_contour(circle_mask.astype(np.uint8))
    
    # Calculate metrics
    metrics = calculate_all_metrics(circle_mask)
    
    # Print baseline results
    print(f"\nBaseline Circle (radius = {radius}):")
    print(f"  Roundness from metrics: {metrics['roundness']:.4f}")
    
    # Visualize contour points
    visualize_contour_points(
        circle_mask, contour, 
        save_path=output_dir / 'circle_contour_points.png'
    )
    
    # Print detailed contour statistics
    print_contour_stats(contour, radius)
    
    # Investigate how roundness varies with radius
    investigate_radius_dependence()
    
    # Check potential fixes
    fix_check()
    
    # Export modified mask
    better_mask = export_modified_mask()
    
    print("\nInvestigation complete. Check the output directory for results.")
