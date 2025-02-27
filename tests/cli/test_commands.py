"""
Tests for the CLI commands.
"""
import os
import pytest
import numpy as np
import tempfile
from pathlib import Path
import argparse
from seg_ana.cli.commands import (
    setup_parser,
    run_analyze_command,
    run_visualize_command,
    run_info_command
)


def create_test_masks(num_masks=5, size=64):
    """Create test masks for testing."""
    masks = np.zeros((num_masks, size, size), dtype=np.uint8)
    
    # Add a large object to each mask
    for i in range(num_masks):
        # Create a circular mask in the center
        center_x, center_y = size // 2, size // 2
        radius = size // 4
        y, x = np.ogrid[:size, :size]
        dist_from_center = np.sqrt((x - center_x)**2 + (y - center_y)**2)
        
        # Large object
        masks[i][dist_from_center <= radius] = 1
    
    return masks


def test_setup_parser():
    """Test argument parser setup."""
    parser = setup_parser()
    assert isinstance(parser, argparse.ArgumentParser)
    
    # Check that expected commands are available
    args = parser.parse_args(['analyze', 'test.npy'])
    assert args.command == 'analyze'
    assert args.input == 'test.npy'
    
    args = parser.parse_args(['visualize', 'test.npy', '--metric', 'area'])
    assert args.command == 'visualize'
    assert args.metric == 'area'
    
    args = parser.parse_args(['info', 'test.npy'])
    assert args.command == 'info'


def test_run_analyze_command():
    """Test the analyze command."""
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create test masks
        masks = create_test_masks()
        
        # Save to temporary file
        temp_npy = os.path.join(temp_dir, 'test_masks.npy')
        np.save(temp_npy, masks)
        
        # Set up arguments
        args = argparse.Namespace(
            input=temp_npy,
            output=os.path.join(temp_dir, 'results.csv'),
            min_area=10,
            threshold=None
        )
        
        # Run command
        run_analyze_command(args)
        
        # Check that output file was created
        assert os.path.exists(args.output)


def test_run_visualize_command():
    """Test the visualize command."""
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create test masks
        masks = create_test_masks()
        
        # Save to temporary file
        temp_npy = os.path.join(temp_dir, 'test_masks.npy')
        np.save(temp_npy, masks)
        
        # Set up arguments
        output_dir = os.path.join(temp_dir, 'viz')
        args = argparse.Namespace(
            input=temp_npy,
            output_dir=output_dir,
            min_area=10,
            threshold=None,
            metric='roundness',
            samples=3,
            grid_samples=4
        )
        
        # Run command
        run_visualize_command(args)
        
        # Check that output directory and files were created
        assert os.path.exists(output_dir)
        assert os.path.exists(os.path.join(output_dir, 'roundness_histogram.png'))
        assert os.path.exists(os.path.join(output_dir, 'roundness_comparison.png'))
        assert os.path.exists(os.path.join(output_dir, 'roundness_grid.png'))


def test_run_info_command(capsys):
    """Test the info command with stdout capture."""
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create test masks
        masks = create_test_masks()
        
        # Save to temporary file
        temp_npy = os.path.join(temp_dir, 'test_masks.npy')
        np.save(temp_npy, masks)
        
        # Set up arguments
        args = argparse.Namespace(
            input=temp_npy,
            min_area=10,
            threshold=None
        )
        
        # Run command
        run_info_command(args)
        
        # Check captured output
        captured = capsys.readouterr()
        assert "Mask Information:" in captured.out
        assert f"Number of masks: {masks.shape[0]}" in captured.out
