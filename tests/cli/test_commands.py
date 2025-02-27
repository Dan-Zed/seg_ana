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
    run_info_command,
    run_generate_command
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
    
    args = parser.parse_args(['info', 'test.npy'])
    assert args.command == 'info'
    
    args = parser.parse_args(['generate', 'test.npy', '--num', '10'])
    assert args.command == 'generate'
    assert args.output == 'test.npy'
    assert args.num == 10


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


def test_run_generate_command():
    """Test the generate command."""
    with tempfile.TemporaryDirectory() as temp_dir:
        # Set up output path
        output_path = os.path.join(temp_dir, 'generated_masks.npy')
        
        # Set up arguments with larger size to avoid low >= high error
        args = argparse.Namespace(
            output=output_path,
            num=3,
            size=256,  # Increased size to avoid padding issue
            shapes='circle',
            seed=42
        )
        
        # Run command
        run_generate_command(args)
        
        # Check that output file was created
        assert os.path.exists(output_path)
        
        # Check that file contains the expected data
        masks = np.load(output_path)
        assert masks.shape[0] == 3  # Number of masks
        assert masks.shape[1] == 256  # Height
        assert masks.shape[2] == 256  # Width
