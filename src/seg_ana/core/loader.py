"""
Module for loading and preprocessing segmentation masks.
"""
import numpy as np
import logging
from pathlib import Path
from typing import Union, Optional, List, Tuple

from skimage.morphology import remove_small_objects

# set up logging
logger = logging.getLogger(__name__)


def load_npy_file(
    npy_path: Union[str, Path]
) -> np.ndarray:
    """
    Load a .npy file containing segmentation masks.
    
    Parameters:
    -----------
    npy_path : str or Path
        Path to the .npy file containing masks
        
    Returns:
    --------
    np.ndarray
        Array of segmentation masks with shape (n_masks, height, width)
        
    Example:
    --------
    >>> masks = load_npy_file('path/to/masks.npy')
    >>> print(masks.shape)
    (10, 512, 512)
    """
    try:
        logger.info(f"Loading masks from {npy_path}")
        path = Path(npy_path)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {npy_path}")
        
        arr = np.load(path)
        logger.info(f"Loaded array with shape {arr.shape}")
        
        # Validate array dimensions
        if arr.ndim < 2:
            raise ValueError(f"Expected at least 2D array, got {arr.ndim}D")
        
        # If only one mask is provided as a 2D array, add a dimension
        if arr.ndim == 2:
            arr = arr[np.newaxis, ...]
            
        return arr
    except Exception as e:
        logger.error(f"Error loading {npy_path}: {str(e)}")
        raise


def preprocess_masks(
    masks: np.ndarray,
    min_area: int = 100,
    binary_threshold: Optional[float] = None
) -> np.ndarray:
    """
    Preprocess segmentation masks by converting to binary and removing small objects.
    
    Parameters:
    -----------
    masks : np.ndarray
        Array of segmentation masks with shape (n_masks, height, width)
    min_area : int, default=100
        Minimum area (in pixels) of objects to keep
    binary_threshold : float, optional
        Threshold for binarization (if masks are not already binary)
        
    Returns:
    --------
    np.ndarray
        Preprocessed binary masks with shape (n_masks, height, width)
        
    Example:
    --------
    >>> raw_masks = load_npy_file('path/to/masks.npy')
    >>> processed = preprocess_masks(raw_masks, min_area=150)
    """
    logger.info(f"Preprocessing {masks.shape[0]} masks")
    
    # Make a copy to avoid modifying the original
    processed = masks.copy()
    
    # Binarize if not already binary and threshold is provided
    if binary_threshold is not None:
        processed = processed > binary_threshold
    else:
        # Convert to boolean if not already
        processed = processed.astype(bool)
    
    # Remove small objects from each mask
    for i in range(processed.shape[0]):
        # Convert to bool to ensure skimage.morphology compatibility
        mask_bool = processed[i].astype(bool)
        processed[i] = remove_small_objects(mask_bool, min_area)
        
    logger.info(f"Preprocessing complete")
    return processed


def load_and_process(
    npy_path: Union[str, Path],
    min_area: int = 100,
    binary_threshold: Optional[float] = None
) -> np.ndarray:
    """
    Load and preprocess segmentation masks in a single function.
    
    Parameters:
    -----------
    npy_path : str or Path
        Path to the .npy file containing masks
    min_area : int, default=100
        Minimum area (in pixels) of objects to keep
    binary_threshold : float, optional
        Threshold for binarization (if masks are not already binary)
        
    Returns:
    --------
    np.ndarray
        Preprocessed binary masks with shape (n_masks, height, width)
        
    Example:
    --------
    >>> processed = load_and_process('path/to/masks.npy', min_area=150)
    """
    masks = load_npy_file(npy_path)
    return preprocess_masks(masks, min_area, binary_threshold)


def get_batch_info(masks: np.ndarray) -> dict:
    """
    Get basic information about a batch of masks.
    
    Parameters:
    -----------
    masks : np.ndarray
        Array of segmentation masks with shape (n_masks, height, width)
        
    Returns:
    --------
    dict
        Dictionary containing basic information about the masks
        
    Example:
    --------
    >>> masks = load_npy_file('path/to/masks.npy')
    >>> info = get_batch_info(masks)
    >>> print(info['num_masks'])
    10
    """
    # Count number of objects in each mask
    object_counts = [np.max(np.unique(m)) for m in masks]
    
    # Calculate total area of objects in each mask
    areas = [np.sum(m > 0) for m in masks]
    
    return {
        'num_masks': masks.shape[0],
        'mask_dimensions': masks.shape[1:],
        'avg_objects_per_mask': np.mean(object_counts),
        'avg_object_area': np.mean(areas),
        'min_object_area': np.min(areas),
        'max_object_area': np.max(areas),
    }
