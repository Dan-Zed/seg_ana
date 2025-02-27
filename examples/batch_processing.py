"""
Batch processing example for the seg-ana package.

This example demonstrates how to:
1. Generate a set of synthetic test shapes
2. Save them as individual .npy files
3. Process them using the batch_analyze script
4. Interpret the results
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import subprocess
import sys

# Import from seg_ana package
from seg_ana.core.synthetic import (
    create_shape_with_protrusions,
    create_mathematical_circle,
    create_ellipse_mask,
    create_circle_mask
)

# Directories
data_dir = Path("./example_data")
output_dir = Path("./example_batch_results")

# Create directories
data_dir.mkdir(exist_ok=True)
output_dir.mkdir(exist_ok=True)


def generate_synthetic_dataset():
    """Generate a variety of shapes and save them as individual .npy files."""
    print("Generating synthetic shapes...")
    
    # Create shapes with different properties
    shapes = {
        "circle": create_mathematical_circle(size=(512, 512), radius=100),
        "ellipse_1_5_ratio": create_ellipse_mask(size=(512, 512), axes=(120, 80)),
        "ellipse_2_1_ratio": create_ellipse_mask(size=(512, 512), axes=(140, 70)),
        "ellipse_3_1_ratio": create_ellipse_mask(size=(512, 512), axes=(150, 50)),
        "protrusions_3": create_shape_with_protrusions(
            size=(512, 512), radius=100, num_protrusions=3, protrusion_size=20
        ),
        "protrusions_6": create_shape_with_protrusions(
            size=(512, 512), radius=100, num_protrusions=6, protrusion_size=20
        ),
        "protrusions_9": create_shape_with_protrusions(
            size=(512, 512), radius=100, num_protrusions=9, protrusion_size=20
        ),
        "noisy_circle_low": create_circle_mask(
            size=(512, 512), radius=100, noise=0.1
        ),
        "noisy_circle_high": create_circle_mask(
            size=(512, 512), radius=100, noise=0.25
        ),
    }
    
    # Save each shape as a separate .npy file
    for name, shape in shapes.items():
        output_path = data_dir / f"{name}.npy"
        np.save(output_path, shape)
        print(f"  Saved {output_path}")
    
    print(f"Generated {len(shapes)} shapes and saved to {data_dir.absolute()}")
    return shapes


def run_batch_analysis():
    """Run the batch_analyze.py script on the generated data."""
    print("\nRunning batch analysis...")
    
    # Get path to the batch_analyze.py script
    script_path = Path("../scripts/batch_analyze.py").resolve()
    
    if not script_path.exists():
        print(f"Error: Could not find {script_path}")
        print("Make sure the scripts directory is in the correct location.")
        return False
    
    # Build and run the command
    cmd = [
        sys.executable,
        str(script_path),
        str(data_dir),
        "--output_dir", str(output_dir),
        "--workers", "2"
    ]
    
    print(f"Running command: {' '.join(cmd)}")
    
    try:
        # Run the process and capture output
        result = subprocess.run(
            cmd, 
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        
        # Print output
        print("\nCommand output:")
        print(result.stdout)
        
        if result.stderr:
            print("\nErrors/warnings:")
            print(result.stderr)
        
        return True
    
    except subprocess.CalledProcessError as e:
        print(f"Error running batch analysis: {e}")
        if e.stdout:
            print("\nOutput:")
            print(e.stdout)
        if e.stderr:
            print("\nErrors:")
            print(e.stderr)
        return False


def explore_results():
    """Look at the batch processing results."""
    print("\nExploring batch processing results...")
    
    # Find the CSV file
    csv_files = list(output_dir.glob("*.csv"))
    
    if not csv_files:
        print("No CSV results found.")
        return
    
    # Use the most recent CSV file
    csv_file = sorted(csv_files)[-1]
    print(f"Found results file: {csv_file}")
    
    try:
        # Load results
        import pandas as pd
        results = pd.read_csv(csv_file)
        
        # Display summary statistics
        print("\nSummary statistics:")
        print(results.describe())
        
        # Plot a comparison of key metrics
        metrics_to_plot = ['roundness', 'ellipticity', 'solidity', 'protrusions']
        
        fig, axes = plt.subplots(len(metrics_to_plot), 1, figsize=(10, 3*len(metrics_to_plot)))
        
        for i, metric in enumerate(metrics_to_plot):
            if metric in results.columns:
                # Sort by metric value
                plot_data = results.sort_values(metric)
                
                # Plot
                axes[i].bar(plot_data['filename'], plot_data[metric])
                axes[i].set_title(f'{metric.capitalize()} by Shape')
                axes[i].set_ylabel(metric)
                axes[i].tick_params(axis='x', rotation=45)
                axes[i].grid(alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(output_dir / "metrics_comparison.png", dpi=150)
        plt.close()
        
        print(f"Created metrics comparison plot: {output_dir / 'metrics_comparison.png'}")
        
    except Exception as e:
        print(f"Error exploring results: {e}")


def main():
    """Run the complete batch processing example."""
    print("Running batch processing example...")
    
    # Step 1: Generate synthetic data
    shapes = generate_synthetic_dataset()
    
    # Step 2: Run batch analysis
    success = run_batch_analysis()
    
    if success:
        # Step 3: Explore the results
        explore_results()
        
        print(f"\nBatch processing example complete.")
        print(f"Input data: {data_dir.absolute()}")
        print(f"Results: {output_dir.absolute()}")
    else:
        print("\nBatch processing failed. Please check the error messages above.")


if __name__ == "__main__":
    main()
