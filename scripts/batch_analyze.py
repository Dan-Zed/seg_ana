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


def safe_format(value, fmt=".1f"):
    """
    Safely format a value, handling both numeric and non-numeric types.
    
    Parameters:
    -----------
    value : any
        The value to format
    fmt : str, default=".1f"
        Format specification for numeric values
        
    Returns:
    --------
    str
        Formatted string representation of the value
    """
    if isinstance(value, (int, float)):
        return f"{value:{fmt}}"
    return str(value)


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
    fig = plt.figure(figsize=(15, 12))  # Increased height for more metrics
    
    # Create layout: 2x3 grid for visualizations plus two text areas
    gs = fig.add_gridspec(3, 4)  # Added an extra row for more metrics
    
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
    
    # 5. Skeleton visualization (new)
    try:
        # Get skeleton
        from skimage.morphology import skeletonize
        skeleton = skeletonize(mask > 0).astype(np.uint8) * 255
        
        ax5 = fig.add_subplot(gs[1, 0])
        ax5.imshow(skeleton, cmap='gray')
        # Safely format the skeleton branches
        branches = metrics.get('skeleton_branches', 'N/A')
        ax5.set_title(f"Skeleton: {branches} branches")
        ax5.axis('off')
    except Exception as e:
        logger.warning(f"Error in skeleton visualization: {str(e)}")
        fig.delaxes(fig.add_subplot(gs[1, 0]))
    
    # 6. Protrusion analysis (if available)
    try:
        # Get body and protrusions
        body_mask, protrusion_masks, visualization = isolate_protrusions(mask)
        
        ax6 = fig.add_subplot(gs[1, 1])
        ax6.imshow(visualization)
        ax6.set_title("Protrusion Analysis")
        ax6.axis('off')
        
        # 7. Protrusions only
        ax7 = fig.add_subplot(gs[1, 2])
        
        # Create a visualization of just the protrusions
        protrusions_only = np.zeros_like(visualization)
        for i, p_mask in enumerate(protrusion_masks):
            # Add each protrusion in a different color
            color = [(i+1) % 3 == 0, (i+1) % 3 == 1, (i+1) % 3 == 2]
            for c in range(3):
                protrusions_only[:, :, c] += p_mask * 255 * color[c]
        
        # Ensure values are in valid range
        protrusions_only = np.clip(protrusions_only, 0, 255)
        
        ax7.imshow(protrusions_only)
        ax7.set_title(f"{len(protrusion_masks)} Protrusions Detected")
        ax7.axis('off')
    except Exception as e:
        logger.warning(f"Error in protrusion analysis visualization: {str(e)}")
        # Remove the subplots if we can't do protrusion analysis
        fig.delaxes(fig.add_subplot(gs[1, 1]))
        fig.delaxes(fig.add_subplot(gs[1, 2]))
    
    # 8. Fractal dimension visualization (new)
    try:
        # Get contour image
        if contours:
            contour_points = contour.squeeze()
            x_min, y_min = np.min(contour_points, axis=0)
            x_max, y_max = np.max(contour_points, axis=0)
            
            width = max(x_max - x_min, 1)
            height = max(y_max - y_min, 1)
            
            contour_img = np.zeros((height, width), dtype=np.uint8)
            adjusted_contour = contour_points - [x_min, y_min]
            cv2.drawContours(contour_img, [adjusted_contour.astype(np.int32)], 0, 255, 1)
            
            ax8 = fig.add_subplot(gs[1, 3])
            ax8.imshow(contour_img, cmap='gray')
            # Safely format the fractal dimension
            fractal_dim = metrics.get('fractal_dimension', 'N/A')
            if isinstance(fractal_dim, (int, float)):
                fractal_text = f"Fractal Dim: {fractal_dim:.4f}"
            else:
                fractal_text = f"Fractal Dim: {fractal_dim}"
                
            ax8.set_title(fractal_text)
            ax8.axis('off')
    except Exception as e:
        logger.warning(f"Error in fractal dimension visualization: {str(e)}")
        # Remove the subplot if we can't visualize fractal dimension
        fig.delaxes(fig.add_subplot(gs[1, 3]))
    
    # Add metrics text box for basic metrics
    ax_basic = fig.add_subplot(gs[2, 0:2])
    ax_basic.axis('off')
    
    # Format basic metrics for display
    basic_metrics_text = f"Basic Metrics for: {mask_name}\n\n"
    
    # Basic shape metrics
    basic_metrics_text += "Basic Shape Metrics:\n"
    basic_metrics_text += f"• Area: {safe_format(metrics.get('area', 'N/A'))} pixels²\n"
    basic_metrics_text += f"• Perimeter: {safe_format(metrics.get('perimeter', 'N/A'))} pixels\n"
    basic_metrics_text += f"• Roundness: {safe_format(metrics.get('roundness', 'N/A'), '.4f')}\n"
    basic_metrics_text += f"• Ellipticity: {safe_format(metrics.get('ellipticity', 'N/A'), '.4f')}\n"
    basic_metrics_text += f"• Solidity: {safe_format(metrics.get('solidity', 'N/A'), '.4f')}\n"
    basic_metrics_text += f"• Convexity: {safe_format(metrics.get('convexity', 'N/A'), '.4f')}\n"
    basic_metrics_text += f"• Protrusion Count: {safe_format(metrics.get('protrusions', 'N/A'), '')}\n"
    basic_metrics_text += f"• Skeleton Complexity: {safe_format(metrics.get('skeleton_complexity', 'N/A'), '.4f')}\n"
    
    ax_basic.text(0, 1, basic_metrics_text, fontsize=11, va='top', 
                 fontfamily='monospace', linespacing=1.5)
    
    # Add metrics text box for experimental metrics
    ax_exp = fig.add_subplot(gs[2, 2:4])
    ax_exp.axis('off')
    
    # Format experimental metrics for display
    exp_metrics_text = f"Experimental Metrics:\n\n"
    
    # Alternative roundness
    exp_metrics_text += f"• Roundness (alt): {safe_format(metrics.get('roundness_alt', 'N/A'), '.4f')}\n"
    exp_metrics_text += f"• Roundness (equiv): {safe_format(metrics.get('roundness_equivalent', 'N/A'), '.4f')}\n"
    
    # Skeleton metrics
    exp_metrics_text += f"• Skeleton Branches: {safe_format(metrics.get('skeleton_branches', 'N/A'), '')}\n"
    exp_metrics_text += f"• Skeleton Branch Length: {safe_format(metrics.get('skeleton_branch_length_mean', 'N/A'))}\n"
    exp_metrics_text += f"• Skeleton Endpoints: {safe_format(metrics.get('skeleton_endpoints', 'N/A'), '')}\n"
    
    # Fractal metrics
    exp_metrics_text += f"• Fractal Dimension: {safe_format(metrics.get('fractal_dimension', 'N/A'), '.4f')}\n"
    exp_metrics_text += f"• Boundary Entropy: {safe_format(metrics.get('boundary_entropy', 'N/A'), '.4f')}\n"
    
    # Protrusion details
    exp_metrics_text += f"• Protrusion Mean Length: {safe_format(metrics.get('protrusion_mean_length', 'N/A'))}\n"
    exp_metrics_text += f"• Protrusion Mean Width: {safe_format(metrics.get('protrusion_mean_width', 'N/A'))}\n"
    exp_metrics_text += f"• Protrusion Length CV: {safe_format(metrics.get('protrusion_length_cv', 'N/A'), '.3f')}\n"
    exp_metrics_text += f"• Protrusion Spacing: {safe_format(metrics.get('protrusion_spacing_uniformity', 'N/A'), '.3f')}\n"
    
    ax_exp.text(0, 1, exp_metrics_text, fontsize=11, va='top', 
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
    
    # Check if we have any valid results
    if df.empty:
        logger.warning("No results to save")
        return df
        
    # Generate timestamp for filenames
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Save all results to CSV first
    full_csv_path = output_dir / f"all_metrics_{timestamp}.csv"
    df.to_csv(full_csv_path, index=False)
    logger.info(f"All results saved to {full_csv_path}")
    
    # Define basic metrics columns we want to extract
    basic_metrics = [
        'filename', 'area', 'perimeter', 'equivalent_diameter',
        'roundness', 'ellipticity', 'solidity', 'convexity',
        'protrusions', 'skeleton_complexity'
    ]
    
    # Filter columns that actually exist
    available_basic_metrics = [col for col in basic_metrics if col in df.columns]
    
    if len(available_basic_metrics) > 1:  # At least filename and one metric
        # Create basic metrics DataFrame with only available columns
        basic_df = df[available_basic_metrics].copy()
        
        # Rename roundness_original to roundness for clarity if needed
        if 'roundness_original' in basic_df.columns and 'roundness' not in basic_df.columns:
            basic_df = basic_df.rename(columns={'roundness_original': 'roundness'})
        
        # Save basic metrics to CSV
        basic_csv_path = output_dir / f"basic_metrics_{timestamp}.csv"
        basic_df.to_csv(basic_csv_path, index=False)
        logger.info(f"Basic metrics saved to {basic_csv_path}")
    else:
        logger.warning("Not enough valid metrics to create basic metrics file")
    
    # Create summary visualizations
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
    if df.empty or 'error' in df.columns and len(df) == len(df['error'].dropna()):
        logger.warning("Cannot create summary visualizations due to errors or empty DataFrame")
        return
    
    output_dir = Path(output_dir)
    
    # Get only numeric columns for visualization
    numeric_df = df.select_dtypes(include=[np.number])
    
    if numeric_df.empty:
        logger.warning("No numeric data available for summary visualizations")
        return
    
    # Create histograms for key basic metrics
    basic_metrics = [
        'roundness', 'roundness_original', 'ellipticity', 'solidity', 'convexity',
        'protrusions', 'skeleton_complexity'
    ]
    
    # Filter to only include columns that exist
    available_basic_metrics = [m for m in basic_metrics if m in numeric_df.columns]
    
    if not available_basic_metrics:
        logger.warning("No basic metrics available for histogram")
    else:
        rows = (len(available_basic_metrics) + 1) // 2  # Calculate needed rows
        fig, axes = plt.subplots(rows, 2, figsize=(14, 5 * rows))
        if rows == 1 and len(available_basic_metrics) == 1:
            axes = np.array([axes])  # Handle single subplot case
        axes = axes.flatten()
        
        for i, metric in enumerate(available_basic_metrics):
            ax = axes[i]
            numeric_df[metric].hist(ax=ax, bins=20, alpha=0.7, color='steelblue')
            title = 'Roundness' if metric == 'roundness_original' else metric.capitalize().replace('_', ' ')
            ax.set_title(f'Distribution of {title}')
            ax.set_xlabel(metric)
            ax.set_ylabel('Count')
            ax.grid(alpha=0.3)
        
        # Remove unused subplots if any
        for j in range(len(available_basic_metrics), len(axes)):
            fig.delaxes(axes[j])
        
        plt.tight_layout()
        plt.savefig(output_dir / f"basic_metrics_histogram_{timestamp}.png", dpi=150)
        plt.close()
    
    # Create histograms for experimental metrics
    experimental_metrics = [
        'roundness_equivalent', 'fractal_dimension', 'boundary_entropy',
        'skeleton_branches', 'protrusion_mean_length', 'protrusion_spacing_uniformity'
    ]
    
    # Filter to only include columns that exist
    available_exp_metrics = [m for m in experimental_metrics if m in numeric_df.columns]
    
    if not available_exp_metrics:
        logger.warning("No experimental metrics available for histogram")
    else:
        rows = (len(available_exp_metrics) + 1) // 2  # Calculate needed rows
        fig, axes = plt.subplots(rows, 2, figsize=(14, 5 * rows))
        
        # Handle case of single subplot
        if rows == 1 and len(available_exp_metrics) == 1:
            axes = np.array([axes])
        axes = axes.flatten()
        
        for i, metric in enumerate(available_exp_metrics):
            ax = axes[i]
            numeric_df[metric].hist(ax=ax, bins=20, alpha=0.7, color='lightgreen')
            # Format title nicely
            title = metric.replace('_', ' ').title()
            ax.set_title(f'Distribution of {title}')
            ax.set_xlabel(metric)
            ax.set_ylabel('Count')
            ax.grid(alpha=0.3)
        
        # Remove unused subplots if any
        for j in range(len(available_exp_metrics), len(axes)):
            fig.delaxes(axes[j])
            
        plt.tight_layout()
        plt.savefig(output_dir / f"experimental_metrics_histogram_{timestamp}.png", dpi=150)
        plt.close()
    
    # Create a correlation matrix if we have enough samples
    if len(numeric_df) >= 5 and len(numeric_df.columns) >= 2:
        # Basic metrics correlation - filter to only include available metrics
        basic_columns = [
            'area', 'perimeter', 'roundness', 'ellipticity',
            'solidity', 'convexity', 'protrusions', 'skeleton_complexity'
        ]
        available_basic_columns = [col for col in basic_columns if col in numeric_df.columns]
        
        if len(available_basic_columns) >= 2:
            basic_corr = numeric_df[available_basic_columns].corr()
            
            # Plot basic correlation matrix
            plt.figure(figsize=(12, 10))
            plt.imshow(basic_corr, cmap='coolwarm', vmin=-1, vmax=1)
            
            # Add correlation values
            for i in range(len(basic_corr.columns)):
                for j in range(len(basic_corr.columns)):
                    plt.text(j, i, f"{basic_corr.iloc[i, j]:.2f}",
                             ha='center', va='center', color='white' if abs(basic_corr.iloc[i, j]) > 0.7 else 'black')
            
            plt.colorbar()
            plt.xticks(range(len(basic_corr.columns)), basic_corr.columns, rotation=45, ha='right')
            plt.yticks(range(len(basic_corr.columns)), basic_corr.columns)
            plt.title('Correlation Matrix of Basic Shape Metrics')
            
            plt.tight_layout()
            plt.savefig(output_dir / f"basic_correlation_matrix_{timestamp}.png", dpi=150)
            plt.close()
        
        # Full correlation matrix - if we have enough metrics
        if len(numeric_df.columns) >= 3:
            # Calculate correlation matrix
            corr_matrix = numeric_df.corr()
            
            # Plot full correlation matrix
            plt.figure(figsize=(16, 14))
            plt.imshow(corr_matrix, cmap='coolwarm', vmin=-1, vmax=1)
            
            # No text labels as there are too many metrics
            plt.colorbar()
            plt.xticks(range(len(corr_matrix.columns)), corr_matrix.columns, rotation=90, fontsize=8)
            plt.yticks(range(len(corr_matrix.columns)), corr_matrix.columns, fontsize=8)
            plt.title('Correlation Matrix of All Metrics')
            
            plt.tight_layout()
            plt.savefig(output_dir / f"full_correlation_matrix_{timestamp}.png", dpi=150)
            plt.close()
    
    # Create key scatter plots
    scatter_pairs = [
        ('roundness', 'solidity'),
        ('ellipticity', 'roundness'),
        ('protrusions', 'solidity'),
        ('skeleton_complexity', 'fractal_dimension')
    ]
    
    for x_metric, y_metric in scatter_pairs:
        if x_metric in numeric_df.columns and y_metric in numeric_df.columns:
            plt.figure(figsize=(10, 8))
            plt.scatter(numeric_df[x_metric], numeric_df[y_metric], alpha=0.7)
            
            # Add labels for outlier points if filename column exists
            if 'filename' in df.columns:
                for i, row in df.iterrows():
                    if pd.isna(row.get(x_metric)) or pd.isna(row.get(y_metric)):
                        continue
                        
                    # Define what makes a point an outlier (can be customized)
                    x_mean, x_std = numeric_df[x_metric].mean(), numeric_df[x_metric].std()
                    y_mean, y_std = numeric_df[y_metric].mean(), numeric_df[y_metric].std()
                    
                    is_outlier = (abs(row[x_metric] - x_mean) > 1.5 * x_std or 
                                abs(row[y_metric] - y_mean) > 1.5 * y_std)
                                
                    if is_outlier:
                        plt.annotate(row['filename'], 
                                    (row[x_metric], row[y_metric]),
                                    textcoords="offset points",
                                    xytext=(0, 5),
                                    ha='center')
            
            plt.xlabel(x_metric.replace('_', ' ').title())
            plt.ylabel(y_metric.replace('_', ' ').title())
            plt.title(f'{y_metric.title()} vs. {x_metric.title()}')
            plt.grid(alpha=0.3)
            
            plt.tight_layout()
            plt.savefig(output_dir / f"{y_metric}_vs_{x_metric}_{timestamp}.png", dpi=150)
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
