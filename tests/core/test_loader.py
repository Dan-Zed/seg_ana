"""
Tests for the loader module.
"""
import numpy as np
import pytest
import tempfile
from pathlib import Path
from seg_ana.core.loader import (
    load_npy_file,
    preprocess_masks,
    load_and_process,
    get_batch_info
)


def create_test_masks(num_masks=5, size=64, with_small_objects=True):
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
        
        # Add small objects if requested
        if with_small_objects:
            # Add a few small objects
            for j in range(3):
                sx = np.random.randint(5, size - 5)
                sy = np.random.randint(5, size - 5)
                small_radius = np.random.randint(1, 3)
                small_dist = np.sqrt((x - sx)**2 + (y - sy)**2)
                masks[i][small_dist <= small_radius] = 1
    
    return masks


def test_load_npy_file():
    """Test loading NPY files."""
    # Create a temporary NPY file
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir) / "test_masks.npy"
        
        # Create and save test masks
        masks = create_test_masks()
        np.save(temp_path, masks)
        
        # Test loading
        loaded = load_npy_file(temp_path)
        assert loaded.shape == masks.shape
        assert np.array_equal(loaded, masks)
        
        # Test loading a single 2D mask
        single_mask = masks[0]
        single_path = Path(temp_dir) / "single_mask.npy"
        np.save(single_path, single_mask)
        
        loaded_single = load_npy_file(single_path)
        assert loaded_single.ndim == 3  # Should add a dimension
        assert loaded_single.shape[0] == 1
        assert np.array_equal(loaded_single[0], single_mask)


def test_preprocess_masks():
    """Test preprocessing masks."""
    # Create masks with small objects
    masks = create_test_masks(with_small_objects=True)
    
    # Count objects before preprocessing
    objects_before = [np.sum(mask > 0) for mask in masks]
    
    # Preprocess masks to remove small objects
    processed = preprocess_masks(masks, min_area=10)
    
    # Count objects after preprocessing
    objects_after = [np.sum(mask > 0) for mask in processed]
    
    # Check that processed masks have fewer pixels (small objects removed)
    for before, after in zip(objects_before, objects_after):
        assert after <= before
    
    # Test with different threshold
    masks_float = masks.astype(float) * 0.5  # Scale to [0, 0.5]
    processed_threshold = preprocess_masks(masks_float, binary_threshold=0.25)
    
    # Check that thresholded masks are binary
    assert np.array_equal(processed_threshold, processed_threshold.astype(bool))


def test_load_and_process():
    """Test combined loading and processing."""
    # Create a temporary NPY file
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir) / "test_masks.npy"
        
        # Create and save test masks
        masks = create_test_masks()
        np.save(temp_path, masks)
        
        # Test loading and processing
        processed = load_and_process(temp_path, min_area=10)
        
        # Check that processed masks are binary
        assert np.array_equal(processed, processed.astype(bool))
        
        # Should have the same number of masks
        assert processed.shape[0] == masks.shape[0]


def test_get_batch_info():
    """Test getting batch information."""
    # Create test masks
    masks = create_test_masks(num_masks=10)
    
    # Get batch info
    info = get_batch_info(masks)
    
    # Check that expected keys exist
    expected_keys = [
        'num_masks', 'mask_dimensions', 'avg_objects_per_mask',
        'avg_object_area', 'min_object_area', 'max_object_area'
    ]
    
    for key in expected_keys:
        assert key in info
    
    # Check specific values
    assert info['num_masks'] == 10
    assert info['mask_dimensions'] == masks.shape[1:]
    assert info['avg_object_area'] > 0


def test_error_handling():
    """Test error handling in loader functions."""
    # Test with non-existent file
    with pytest.raises(FileNotFoundError):
        load_npy_file("nonexistent_file.npy")
    
    # Test with invalid array dimensions
    with pytest.raises(ValueError):
        preprocess_masks(np.array([1, 2, 3]))  # 1D array
