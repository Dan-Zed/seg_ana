"""
Batch analysis script for processing multiple mask files.

This script processes all *.npy mask files in a specified directory,
calculates metrics, and generates visualizations and a CSV summary.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2
from pathlib import Path
import argparse
import logging
from datetime import datetime
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing

from seg_ana.core.metrics_improved import calculate_all_metrics
from seg_ana.core.protrusion_analysis import (
    analyze_all_protrusions, 
    summarize_protrusions,
    isolate_protrusions
)

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def visualize_mask_analysis(mask, metrics, output_path):
    """
    Create a comprehensive visualization of a mask and its metrics.
    
    Parameters:
    -----------
    mask : np.ndarray
        The input binary mask
    metrics : dict
        Dictionary of calculated metrics
    output_path : str or Path
        Path to save the visualization
    """
    # Create a figure with subplots
    fig = plt.figure(figsize=(15, 10))
    
    # Create layout: 2x3 grid for visualizations plus a text area
    gs = fig.add_gridspec(2, 4)
    
    # Extract mask name from output path
    mask_name = Path(output_path).stem
    
    # 1. Original mask
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.imshow(mask, cmap='gray')
    ax1.set_title("Original Mask")
    ax1.axis('off')
    
    # Get contour and hull for visualization
    contours, _ = cv2.findContours(
        mask.astype(np.uint8), 
        cv2.RETR_EXTERNAL, 
        cv2.CHAIN_APPROX_SIMPLE
    )
    if contours:
        contour = max(contours, key=cv2.contourArea)
        hull = cv2.convexHull(contour)
        
        # 2. Contour visualization
        ax2 = fig.add_subplot(gs[0, 1])
        ax2.imshow(mask, cmap='gray')
        
        # Draw contour
        contour_points = contour.squeeze()
        ax2.plot(contour_points[:, 0], contour_points[:, 1], 'r-', linewidth=2)
        
        ax2.set_title("Contour")
        ax2.axis('off')
        
        # 3. Convex hull visualization
        ax3 = fig.add_subplot(gs[0, 2])
        ax3.imshow(mask, cmap='gray')
        
        # Draw contour and hull
        hull_points = hull.squeeze()
        ax3.plot(contour_points[:, 0], contour_points[:, 1], 'r-', linewidth=2)
        ax3.plot(hull_points[:, 0], hull_points[:, 1], 'g-', linewidth=2)
        
        ax3.set_title("Convex Hull")
        ax3.axis('off')
        
        # Try to get an ellipse fit if there are enough points
        if len(contour) >= 5:
            # 4. Ellipse fit visualization
            ax4 = fig.add_subplot(gs[0, 3])
            ax4.imshow(mask, cmap='gray')
            
            # Draw contour
            ax4.plot(contour_points[:, 0], contour_points[:, 1], 'r-', linewidth=2)
            
            # Fit and draw ellipse
            ellipse = cv2.fitEllipse(contour)
            center, axes, angle = ellipse
            
            from matplotlib.patches import Ellipse
            e = Ellipse(
                xy=center, width=axes[0], height=axes[1],
                angle=angle, fill=False, edgecolor='blue', linewidth=2
            )
            ax4.add_patch(e)
            
            ax4.set_title("Ellipse Fit")
            ax4.axis('off')
        else:
            # Remove the subplot if we can't fit an ellipse
            fig.delaxes(fig.add_subplot(gs[0, 3]))
    
    # 5. Protrusion analysis (if available)
    try:
        # Get body and protrusions
        body_mask, protrusion_masks, visualization = isolate_protrusions(mask)
        
        ax5 = fig.add_subplot(gs[1, 0])
        ax5.imshow(visualization)
        ax5.set_title("Protrusion Analysis")
        ax5.axis('off')
        
        # 6. Protrusions only
        ax6 = fig.add_subplot(gs[1, 1])
        
        # Create a visualization of just the protrusions
        protrusions_only = np.zeros_like(visualization)
        for i, p_mask in enumerate(protrusion_masks):
            # Add each protrusion in a different color
            color = [(i+1) % 3 == 0, (i+1) % 3 == 1, (i+1) % 3 == 2]
            for c in range(3):
                protrusions_only[:, :, c] += p_mask * 255 * color[c]
        
        # Ensure values are in valid range
        protrusions_only = np.clip(protrusions_only, 0, 255)
        
        ax6.imshow(protrusions_only)
        ax6.set_title(f"{len(protrusion_masks)} Protrusions Detected")
        ax6.axis('off')
    except Exception as e:
        logger.warning(f"Error in protrusion analysis visualization: {str(e)}")
        # Remove the subplots if we can't do protrusion analysis
        fig.delaxes(fig.add_subplot(gs[1, 0]))
        fig.delaxes(fig.add_subplot(gs[1, 1]))
    
    # Add metrics text box spanning the remaining area
    ax_text = fig.add_subplot(gs[1, 2:])
    ax_text.axis('off')
    
    # Format metrics for display
    metrics_text = f"Metrics for: {mask_name}\n\n"
    
    # Basic shape metrics
    metrics_text += "Basic Shape Metrics:\n"
    metrics_text += f"• Area: {metrics.get('area', 'N/A'):.1f} pixels²\n"
    metrics_text += f"• Perimeter: {metrics.get('perimeter', 'N/A'):.1f} pixels\n"
    metrics_text += f"• Roundness: {metrics.get('roundness', 'N/A'):.4f}\n"
    metrics_text += f"• Equivalent Diameter: {metrics.get('equivalent_diameter', 'N/A'):.1f} pixels\n\n"
    
    # Ellipse metrics
    metrics_text += "Ellipse Metrics:\n"
    metrics_text += f"• Ellipticity: {metrics.get('ellipticity', 'N/A'):.4f}\n"
    metrics_text += f"• Major Axis: {metrics.get('major_axis', 'N/A'):.1f} pixels\n"
    metrics_text += f"• Minor Axis: {metrics.get('minor_axis', 'N/A'):.1f} pixels\n"
    metrics_text += f"• Orientation: {metrics.get('orientation', 'N/A'):.1f}°\n\n"
    
    # Convexity metrics
    metrics_text += "Convexity Metrics:\n"
    metrics_text += f"• Solidity: {metrics.get('solidity', 'N/A'):.4f}\n"
    metrics_text += f"• Convexity: {metrics.get('convexity', 'N/A'):.4f}\n\n"
    
    # Protrusion metrics
    metrics_text += "Protrusion Metrics:\n"
    metrics_text += f"• Protrusion Count: {metrics.get('protrusions', 'N/A')}\n"
    
    # Add enhanced protrusion metrics if available
    if 'protrusion_mean_length' in metrics:
        metrics_text += f"• Mean Protrusion Length: {metrics.get('protrusion_mean_length', 'N/A'):.1f} pixels\n"
        metrics_text += f"• Mean Protrusion Width: {metrics.get('protrusion_mean_width', 'N/A'):.1f} pixels\n"
        metrics_text += f"• Length Variation (CV): {metrics.get('protrusion_length_cv', 'N/A'):.3f}\n"
        metrics_text += f"• Spacing Uniformity: {metrics.get('protrusion_spacing_uniformity', 'N/A'):.3f}\n"
    
    ax_text.text(0, 1, metrics_text, fontsize=11, va='top', 
                 fontfamily='monospace', linespacing=1.5)
    
    # Add overall title
    fig.suptitle(f"Shape Analysis: {mask_name}", fontsize=16, y=0.98)
    
    # Save the figure
    plt.tight_layout(rect=[0, 0, 1, 0.96])  # Adjust for the suptitle
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


def process_mask(mask_path, output_dir):
    """
    Process a single mask file and return its metrics.
    
    Parameters:
    -----------
    mask_path : str or Path
        Path to the mask file
    output_dir : str or Path
        Directory to save outputs
        
    Returns:
    --------
    dict
        Dictionary of calculated metrics with filename
    """
    try:
        # Load mask
        mask = np.load(mask_path)
        
        # Ensure binary mask
        if mask.dtype != bool and mask.dtype != np.uint8:
            mask = mask.astype(bool).astype(np.uint8)
        
        # Extract filename without extension
        filename = Path(mask_path).stem
        
        # Calculate metrics
        metrics = calculate_all_metrics(mask)
        
        # Save visualization
        output_path = Path(output_dir) / f"{filename}_analysis.png"
        visualize_mask_analysis(mask, metrics, output_path)
        
        # Add filename to metrics for CSV output
        metrics['filename'] = filename
        
        return metrics
    
    except Exception as e:
        logger.error(f"Error processing {mask_path}: {str(e)}")
        return {'filename': Path(mask_path).stem, 'error': str(e)}


def batch_process(input_dir, output_dir, max_workers=None):
    """
    Process all .npy files in the input directory and save results.
    
    Parameters:
    -----------
    input_dir : str or Path
        Directory containing .npy mask files
    output_dir : str or Path
        Directory to save outputs
    max_workers : int, optional
        Maximum number of worker processes to use
        
    Returns:
    --------
    pandas.DataFrame
        DataFrame containing all calculated metrics
    """
    # Create output directory if it doesn't exist
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)
    
    # Find all .npy files in the input directory
    input_dir = Path(input_dir)
    mask_files = list(input_dir.glob("*.npy"))
    
    if not mask_files:
        logger.warning(f"No .npy files found in {input_dir}")
        return pd.DataFrame()
    
    logger.info(f"Found {len(mask_files)} mask files to process")
    
    # Set default max_workers if not specified
    if max_workers is None:
        max_workers = min(multiprocessing.cpu_count(), len(mask_files))
    
    # Process masks in parallel
    results = []
    
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        # Submit all jobs
        future_to_path = {
            executor.submit(process_mask, str(path), str(output_dir)): path
            for path in mask_files
        }
        
        # Process as they complete
        for i, future in enumerate(as_completed(future_to_path), 1):
            path = future_to_path[future]
            try:
                metrics = future.result()
                results.append(metrics)
                logger.info(f"Processed {i}/{len(mask_files)}: {path.name}")
            except Exception as e:
                logger.error(f"Error processing {path.name}: {str(e)}")
    
    # Convert results to DataFrame
    df = pd.DataFrame(results)
    
    # Save to CSV
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = output_dir / f"metrics_summary_{timestamp}.csv"
    df.to_csv(csv_path, index=False)
    
    logger.info(f"Results saved to {csv_path}")
    
    # Create a summary visualization
    create_summary_visualization(df, output_dir, timestamp)
    
    return df


def create_summary_visualization(df, output_dir, timestamp):
    """
    Create summary visualizations of the batch results.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame containing metrics for all processed masks
    output_dir : str or Path
        Directory to save outputs
    timestamp : str
        Timestamp string for filenames
    """
    if df.empty or 'error' in df.columns:
        logger.warning("Cannot create summary visualizations due to errors or empty DataFrame")
        return
    
    output_dir = Path(output_dir)
    
    # Create histograms for key metrics
    metrics_to_plot = [
        'roundness', 'ellipticity', 'solidity', 'protrusions'
    ]
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()
    
    for i, metric in enumerate(metrics_to_plot):
        if metric in df.columns:
            ax = axes[i]
            df[metric].hist(ax=ax, bins=20, alpha=0.7, color='steelblue')
            ax.set_title(f'Distribution of {metric.capitalize()}')
            ax.set_xlabel(metric)
            ax.set_ylabel('Count')
            ax.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / f"metrics_histogram_{timestamp}.png", dpi=150)
    plt.close()
    
    # Create a correlation matrix if we have enough samples
    if len(df) >= 5:
        # Select only numeric columns
        numeric_df = df.select_dtypes(include=[np.number])
        
        # Calculate correlation matrix
        corr_matrix = numeric_df.corr()
        
        # Plot correlation matrix
        plt.figure(figsize=(12, 10))
        plt.imshow(corr_matrix, cmap='coolwarm', vmin=-1, vmax=1)
        
        # Add correlation values
        for i in range(len(corr_matrix.columns)):
            for j in range(len(corr_matrix.columns)):
                plt.text(j, i, f"{corr_matrix.iloc[i, j]:.2f}",
                         ha='center', va='center', color='white' if abs(corr_matrix.iloc[i, j]) > 0.7 else 'black')
        
        plt.colorbar()
        plt.xticks(range(len(corr_matrix.columns)), corr_matrix.columns, rotation=45, ha='right')
        plt.yticks(range(len(corr_matrix.columns)), corr_matrix.columns)
        plt.title('Correlation Matrix of Shape Metrics')
        
        plt.tight_layout()
        plt.savefig(output_dir / f"correlation_matrix_{timestamp}.png", dpi=150)
        plt.close()
    
    # Create a scatter plot of roundness vs. solidity
    if 'roundness' in df.columns and 'solidity' in df.columns:
        plt.figure(figsize=(10, 8))
        plt.scatter(df['roundness'], df['solidity'], alpha=0.7)
        
        # Add labels for outlier points
        for i, row in df.iterrows():
            if row['roundness'] < 0.7 or row['solidity'] < 0.8:
                plt.annotate(row['filename'], 
                             (row['roundness'], row['solidity']),
                             textcoords="offset points",
                             xytext=(0, 5),
                             ha='center')
        
        plt.xlabel('Roundness')
        plt.ylabel('Solidity')
        plt.title('Roundness vs. Solidity')
        plt.grid(alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(output_dir / f"roundness_vs_solidity_{timestamp}.png", dpi=150)
        plt.close()


def main():
    """Main function to parse arguments and run batch processing."""
    parser = argparse.ArgumentParser(
        description="Batch process mask files to calculate shape metrics"
    )
    
    parser.add_argument(
        "input_dir",
        help="Directory containing .npy mask files"
    )
    
    parser.add_argument(
        "--output_dir",
        default="./batch_results",
        help="Directory to save outputs (default: ./batch_results)"
    )
    
    parser.add_argument(
        "--workers",
        type=int,
        default=None,
        help="Maximum number of worker processes (default: number of CPU cores)"
    )
    
    args = parser.parse_args()
    
    logger.info(f"Starting batch processing from {args.input_dir}")
    logger.info(f"Outputs will be saved to {args.output_dir}")
    
    # Run batch processing
    results = batch_process(args.input_dir, args.output_dir, args.workers)
    
    if not results.empty:
        logger.info(f"Successfully processed {len(results)} mask files")
    else:
        logger.warning("No results were obtained")


if __name__ == "__main__":
    main()
