"""
Command-line interface for segmentation analysis.
"""
import os
import sys
import argparse
import logging
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Optional, List, Dict, Union, Tuple

from ..core.loader import load_and_process, get_batch_info
from ..core.metrics import analyze_batch
from ..core.synthetic import save_test_dataset

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def setup_parser() -> argparse.ArgumentParser:
    """
    Set up command-line argument parser.
    
    Returns:
    --------
    argparse.ArgumentParser
        Configured argument parser
    """
    parser = argparse.ArgumentParser(
        description='Analyze segmentation masks for morphological features'
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Command to run')
    
    # Analyze command
    analyze_parser = subparsers.add_parser('analyze', help='Analyze masks and export metrics')
    analyze_parser.add_argument('input', type=str, help='Path to .npy file with masks')
    analyze_parser.add_argument(
        '--output', '-o', type=str, default=None,
        help='Path to output CSV file (default: derived from input filename)'
    )
    analyze_parser.add_argument(
        '--min-area', type=int, default=100,
        help='Minimum area (in pixels) for objects (default: 100)'
    )
    analyze_parser.add_argument(
        '--threshold', type=float, default=None,
        help='Threshold for binary mask creation (if not already binary)'
    )
    
    # Info command
    info_parser = subparsers.add_parser('info', help='Show information about masks')
    info_parser.add_argument('input', type=str, help='Path to .npy file with masks')
    info_parser.add_argument(
        '--min-area', type=int, default=100,
        help='Minimum area (in pixels) for objects (default: 100)'
    )
    info_parser.add_argument(
        '--threshold', type=float, default=None,
        help='Threshold for binary mask creation (if not already binary)'
    )
    
    # Generate command
    gen_parser = subparsers.add_parser('generate', help='Generate synthetic test data')
    gen_parser.add_argument(
        'output', type=str, help='Path to output .npy file for generated masks'
    )
    gen_parser.add_argument(
        '--num', '-n', type=int, default=100,
        help='Number of masks to generate (default: 100)'
    )
    gen_parser.add_argument(
        '--size', type=int, default=512,
        help='Size of masks (width and height in pixels, default: 512)'
    )
    gen_parser.add_argument(
        '--shapes', type=str, default='all',
        help='Comma-separated list of shape types to include (circle,ellipse,protrusions,random) or "all" (default: all)'
    )
    gen_parser.add_argument(
        '--seed', type=int, default=None,
        help='Random seed for reproducibility (default: None)'
    )
    
    return parser


def run_analyze_command(args: argparse.Namespace) -> None:
    """
    Run the 'analyze' command to calculate metrics and export to CSV.
    
    Parameters:
    -----------
    args : argparse.Namespace
        Command-line arguments
    """
    try:
        logger.info(f"Loading and processing masks from {args.input}")
        masks = load_and_process(
            args.input,
            min_area=args.min_area,
            binary_threshold=args.threshold
        )
        
        logger.info(f"Analyzing {masks.shape[0]} masks")
        metrics_list = analyze_batch(masks)
        
        # Convert to DataFrame
        df = pd.DataFrame(metrics_list)
        
        # Determine output path if not specified
        if args.output is None:
            input_path = Path(args.input)
            output_path = input_path.with_suffix('.csv')
        else:
            output_path = Path(args.output)
        
        # Ensure output directory exists
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Save to CSV
        logger.info(f"Saving metrics to {output_path}")
        df.to_csv(output_path, index=False)
        
        # Print summary
        print(f"\nAnalysis complete for {masks.shape[0]} masks")
        print(f"Results saved to: {output_path}")
        print("\nMetrics summary:")
        print(df.describe().to_string())
        
    except Exception as e:
        logger.error(f"Error in analyze command: {str(e)}")
        sys.exit(1)


def run_info_command(args: argparse.Namespace) -> None:
    """
    Run the 'info' command to display information about masks.
    
    Parameters:
    -----------
    args : argparse.Namespace
        Command-line arguments
    """
    try:
        logger.info(f"Loading and processing masks from {args.input}")
        masks = load_and_process(
            args.input,
            min_area=args.min_area,
            binary_threshold=args.threshold
        )
        
        # Get basic info
        info = get_batch_info(masks)
        
        # Print info
        print("\nMask Information:")
        print(f"Input file: {args.input}")
        print(f"Number of masks: {info['num_masks']}")
        print(f"Mask dimensions: {info['mask_dimensions'][0]} x {info['mask_dimensions'][1]}")
        print(f"Average objects per mask: {info['avg_objects_per_mask']:.2f}")
        print(f"Average object area: {info['avg_object_area']:.2f} pixels")
        print(f"Min object area: {info['min_object_area']:.2f} pixels")
        print(f"Max object area: {info['max_object_area']:.2f} pixels")
        
        # Calculate memory usage
        mem_bytes = masks.nbytes
        mem_mb = mem_bytes / (1024 * 1024)
        print(f"\nMemory usage: {mem_mb:.2f} MB")
        
    except Exception as e:
        logger.error(f"Error in info command: {str(e)}")
        sys.exit(1)


def run_generate_command(args: argparse.Namespace) -> None:
    """
    Run the 'generate' command to create synthetic test data.
    
    Parameters:
    -----------
    args : argparse.Namespace
        Command-line arguments
    """
    try:
        # Parse shape types
        if args.shapes.lower() == 'all':
            shape_types = None  # Default to all shapes
        else:
            shape_types = args.shapes.split(',')
            valid_types = ['circle', 'ellipse', 'protrusions', 'random']
            for shape in shape_types:
                if shape not in valid_types:
                    raise ValueError(
                        f"Invalid shape type: {shape}. "
                        f"Valid types are: {', '.join(valid_types)}"
                    )
        
        # Generate and save masks
        logger.info(f"Generating {args.num} synthetic masks")
        save_test_dataset(
            args.output,
            n_masks=args.num,
            size=(args.size, args.size),
            shape_types=shape_types,
            random_seed=args.seed
        )
        
        print(f"\nGenerated {args.num} synthetic masks")
        print(f"Saved to: {args.output}")
        print(f"Mask size: {args.size}x{args.size}")
        if shape_types:
            print(f"Shape types: {', '.join(shape_types)}")
        else:
            print("Shape types: all (circle, ellipse, protrusions, random)")
        
    except Exception as e:
        logger.error(f"Error in generate command: {str(e)}")
        sys.exit(1)


def main() -> None:
    """
    Main entry point for the CLI.
    """
    parser = setup_parser()
    args = parser.parse_args()
    
    if args.command is None:
        parser.print_help()
        sys.exit(0)
    
    if args.command == 'analyze':
        run_analyze_command(args)
    elif args.command == 'info':
        run_info_command(args)
    elif args.command == 'generate':
        run_generate_command(args)
