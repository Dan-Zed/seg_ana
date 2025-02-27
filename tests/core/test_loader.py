"""
Tests for the loader module.
"""
import numpy as np
import pytest
import tempfile
from seg_ana.core.loader import (
    load_npy_file,
    preprocess_masks,
    load_and_process,
    get_batch_info
)


def create_test_array(shape=(5, 64, 64)):
    """Create a test array for testing."""
    arr = np.zeros(shape, dtype=np.uint8)
    
    # Add some objects to each mask
    for i in range(shape[0]):
        # Create a circular mask in the center
        center_x, center_y = shape[1] // 2, shape[2] // 2
        radius = shape[1] // 4
        y, x = np.ogrid[:shape[1], :shape[2]]
        dist_from_center = np.sqrt((x - center_x)**2 + (y - center_y)**2)
        
        # Large object
        arr[i][dist_from_center <= radius] = 1
    
    return arr


def create_array_with_small_objects(shape=(3, 64, 64)):
    """Create a test array with controlled small objects."""
    arr = np.zeros(shape, dtype=np.uint8)
    
    # For each mask
    for i in range(shape[0]):
        # Add main object
        arr[i, 20:40, 20:40] = 1  # 20x20 square (area = 400)
        
        # Add small object (3x3 square, area = 9)
        arr[i, 50:53, 50:53] = 1
    
    return arr


def test_load_npy_file():
    """Test loading a .npy file."""
    with tempfile.NamedTemporaryFile(suffix='.npy') as temp_file:
        # Create and save a test array
        arr = create_test_array()
        np.save(temp_file.name, arr)
        
        # Load the array
        loaded = load_npy_file(temp_file.name)
        
        # Check if the loaded array is the same as the original
        assert loaded.shape == arr.shape
        assert np.array_equal(loaded, arr)
        
        # Test with single mask (2D)
        single_mask = arr[0]
        np.save(temp_file.name, single_mask)
        loaded_single = load_npy_file(temp_file.name)
        
        # Should add batch dimension
        assert loaded_single.shape == (1,) + single_mask.shape
        assert np.array_equal(loaded_single[0], single_mask)


def test_preprocess_masks():
    """Test preprocessing masks."""
    # Create test array with small objects
    arr = create_array_with_small_objects()
    
    # Calculate original areas
    orig_areas = [np.sum(arr[i] > 0) for i in range(arr.shape[0])]
    
    # Expected area of main object without small objects
    expected_area = 400  # 20x20 square
    
    # Preprocess masks with min_area=10 (should remove 3x3 objects of area 9)
    processed = preprocess_masks(arr, min_area=10)
    
    # Check if the output is binary
    assert processed.dtype == bool
    
    # Check if small objects were removed
    for i in range(arr.shape[0]):
        # Check original has both objects
        assert orig_areas[i] == 409, f"Original should have area 409 (400+9), got {orig_areas[i]}"
        
        # Check processed has only the main object
        area_proc = np.sum(processed[i])
        assert area_proc == expected_area, f"Expected {expected_area}, got {area_proc}"
        
        # Make sure area after processing is smaller than original
        assert area_proc < orig_areas[i], f"Small objects not removed: orig={orig_areas[i]}, processed={area_proc}"


def test_load_and_process():
    """Test loading and processing in one step."""
    with tempfile.NamedTemporaryFile(suffix='.npy') as temp_file:
        # Create test array with small objects
        arr = create_array_with_small_objects()
        
        # Expected area after small objects removed
        expected_area = 400  # 20x20 square
        
        np.save(temp_file.name, arr)
        
        # Load and process
        processed = load_and_process(temp_file.name, min_area=10)
        
        # Check if the output is binary
        assert processed.dtype == bool
        
        # Check if the output has the correct shape
        assert processed.shape == arr.shape
        
        # Check if small objects were removed
        for i in range(arr.shape[0]):
            # Original area (main object + small object)
            orig_area = np.sum(arr[i] > 0)
            assert orig_area == 409, f"Original should have area 409 (400+9), got {orig_area}"
            
            # Processed area (should only have main object)
            proc_area = np.sum(processed[i])
            assert proc_area == expected_area, f"Expected {expected_area}, got {proc_area}"
            
            # Make sure area is smaller after processing
            assert proc_area < orig_area, f"Small objects not removed: orig={orig_area}, processed={proc_area}"


def test_get_batch_info():
    """Test getting batch info."""
    # Create test array
    arr = create_test_array(shape=(3, 64, 64))
    
    # Get batch info
    info = get_batch_info(arr)
    
    # Check if all expected info is present
    assert 'num_masks' in info
    assert 'mask_dimensions' in info
    assert 'avg_objects_per_mask' in info
    assert 'avg_object_area' in info
    assert 'min_object_area' in info
    assert 'max_object_area' in info
    
    # Check values
    assert info['num_masks'] == arr.shape[0]
    assert info['mask_dimensions'] == arr.shape[1:]
    
    # Basic sanity checks
    assert info['avg_object_area'] > 0
    assert info['min_object_area'] <= info['max_object_area']


def test_error_handling():
    """Test error handling in loader functions."""
    # Test with non-existent file
    with pytest.raises(FileNotFoundError):
        load_npy_file("nonexistent_file.npy")

    # Test with invalid array dimensions
    with pytest.raises(ValueError):
        # Create a 1D array to trigger the error
        preprocess_masks(np.array([1, 2, 3]))
