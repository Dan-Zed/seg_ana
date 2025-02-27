"""
Module for generating synthetic data for testing and demonstrations.
"""
import numpy as np
import cv2
import logging
from typing import Tuple, List, Optional

# set up logging
logger = logging.getLogger(__name__)


def create_circle_mask(
    size: Tuple[int, int] = (512, 512),
    center: Optional[Tuple[int, int]] = None,
    radius: int = 50,
    noise: float = 0.0
) -> np.ndarray:
    """
    Create a circular binary mask.
    
    Parameters:
    -----------
    size : tuple, default=(512, 512)
        Size of the mask (height, width)
    center : tuple, optional
        Center coordinates (x, y). If None, uses the center of the image.
    radius : int, default=50
        Radius of the circle
    noise : float, default=0.0
        Amount of noise to add to the circle boundary (0.0 to 1.0)
        
    Returns:
    --------
    np.ndarray
        Binary mask with a circle
        
    Example:
    --------
    >>> mask = create_circle_mask(size=(100, 100), radius=30)
    """
    height, width = size
    
    if center is None:
        center = (width // 2, height // 2)
    
    # Create empty mask
    mask = np.zeros(size, dtype=np.uint8)
    
    # Draw circle
    cv2.circle(mask, center, radius, 1, -1)
    
    # Add noise if requested
    if noise > 0:
        # Generate noise around the perimeter
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        if contours:
            contour = contours[0]
            
            # Add random displacements to contour points
            for i in range(len(contour)):
                # Random displacement
                dx = int(np.random.normal(0, noise * radius / 2))
                dy = int(np.random.normal(0, noise * radius / 2))
                
                # Apply displacement
                contour[i][0][0] += dx
                contour[i][0][1] += dy
            
            # Create new mask from noisy contour
            noise_mask = np.zeros_like(mask)
            cv2.drawContours(noise_mask, [contour], 0, 1, -1)
            mask = noise_mask
    
    return mask


def create_ellipse_mask(
    size: Tuple[int, int] = (512, 512),
    center: Optional[Tuple[int, int]] = None,
    axes: Tuple[int, int] = (80, 40),
    angle: float = 0.0,
    noise: float = 0.0
) -> np.ndarray:
    """
    Create an elliptical binary mask.
    
    Parameters:
    -----------
    size : tuple, default=(512, 512)
        Size of the mask (height, width)
    center : tuple, optional
        Center coordinates (x, y). If None, uses the center of the image.
    axes : tuple, default=(80, 40)
        Semi-major and semi-minor axes lengths (a, b)
    angle : float, default=0.0
        Rotation angle in degrees
    noise : float, default=0.0
        Amount of noise to add to the ellipse boundary (0.0 to 1.0)
        
    Returns:
    --------
    np.ndarray
        Binary mask with an ellipse
        
    Example:
    --------
    >>> mask = create_ellipse_mask(axes=(100, 50), angle=45)
    """
    height, width = size
    
    if center is None:
        center = (width // 2, height // 2)
    
    # Create empty mask
    mask = np.zeros(size, dtype=np.uint8)
    
    # Draw ellipse
    cv2.ellipse(mask, center, axes, angle, 0, 360, 1, -1)
    
    # Add noise if requested
    if noise > 0:
        # Similar approach as in create_circle_mask
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        if contours:
            contour = contours[0]
            
            # Add random displacements to contour points
            max_axis = max(axes)
            for i in range(len(contour)):
                # Random displacement
                dx = int(np.random.normal(0, noise * max_axis / 2))
                dy = int(np.random.normal(0, noise * max_axis / 2))
                
                # Apply displacement
                contour[i][0][0] += dx
                contour[i][0][1] += dy
            
            # Create new mask from noisy contour
            noise_mask = np.zeros_like(mask)
            cv2.drawContours(noise_mask, [contour], 0, 1, -1)
            mask = noise_mask
    
    return mask


def create_shape_with_protrusions(
    size: Tuple[int, int] = (512, 512),
    center: Optional[Tuple[int, int]] = None,
    radius: int = 50,
    num_protrusions: int = 4,
    protrusion_size: int = 10,
    protrusion_distance: float = 1.2,
    random_seed: Optional[int] = None
) -> np.ndarray:
    """
    Create a shape with protrusions (bumps) around the perimeter.
    
    Parameters:
    -----------
    size : tuple, default=(512, 512)
        Size of the mask (height, width)
    center : tuple, optional
        Center coordinates (x, y). If None, uses the center of the image.
    radius : int, default=50
        Base radius of the shape
    num_protrusions : int, default=4
        Number of protrusions to add
    protrusion_size : int, default=10
        Size (radius) of each protrusion
    protrusion_distance : float, default=1.2
        Distance of protrusions from center, as a factor of radius
    random_seed : int, optional
        Random seed for reproducibility
        
    Returns:
    --------
    np.ndarray
        Binary mask with a shape having protrusions
        
    Example:
    --------
    >>> mask = create_shape_with_protrusions(num_protrusions=6)
    """
    height, width = size
    
    if center is None:
        center = (width // 2, height // 2)
    
    # Set random seed if provided
    if random_seed is not None:
        np.random.seed(random_seed)
    
    # Create base shape (circle)
    mask = np.zeros(size, dtype=np.uint8)
    cv2.circle(mask, center, radius, 1, -1)
    
    # Add protrusions
    for i in range(num_protrusions):
        # Calculate angle and position
        angle = i * (2 * np.pi / num_protrusions)
        x = int(center[0] + radius * protrusion_distance * np.cos(angle))
        y = int(center[1] + radius * protrusion_distance * np.sin(angle))
        
        # Add a circle at the protrusion position
        cv2.circle(mask, (x, y), protrusion_size, 1, -1)
    
    return mask


def create_random_shape(
    size: Tuple[int, int] = (512, 512),
    center: Optional[Tuple[int, int]] = None,
    max_vertices: int = 12,
    min_vertices: int = 6,
    radius_range: Tuple[int, int] = (30, 70),
    random_seed: Optional[int] = None
) -> np.ndarray:
    """
    Create a random polygon shape.
    
    Parameters:
    -----------
    size : tuple, default=(512, 512)
        Size of the mask (height, width)
    center : tuple, optional
        Center coordinates (x, y). If None, uses the center of the image.
    max_vertices : int, default=12
        Maximum number of vertices for the polygon
    min_vertices : int, default=6
        Minimum number of vertices for the polygon
    radius_range : tuple, default=(30, 70)
        Range of distances from center to vertices (min, max)
    random_seed : int, optional
        Random seed for reproducibility
        
    Returns:
    --------
    np.ndarray
        Binary mask with a random polygon shape
        
    Example:
    --------
    >>> mask = create_random_shape(min_vertices=8, max_vertices=16)
    """
    height, width = size
    
    if center is None:
        center = (width // 2, height // 2)
    
    # Set random seed if provided
    if random_seed is not None:
        np.random.seed(random_seed)
    
    # Determine number of vertices
    num_vertices = np.random.randint(min_vertices, max_vertices + 1)
    
    # Generate vertices around the center
    vertices = []
    for i in range(num_vertices):
        # Angle with some randomness
        angle = i * (2 * np.pi / num_vertices) + np.random.uniform(-0.2, 0.2)
        
        # Distance from center (radius)
        r = np.random.uniform(radius_range[0], radius_range[1])
        
        # Calculate vertex position
        x = int(center[0] + r * np.cos(angle))
        y = int(center[1] + r * np.sin(angle))
        
        vertices.append([x, y])
    
    # Create mask with polygon
    mask = np.zeros(size, dtype=np.uint8)
    vertices = np.array(vertices, dtype=np.int32)
    cv2.fillPoly(mask, [vertices], 1)
    
    return mask


def create_test_dataset(
    n_masks: int = 100,
    size: Tuple[int, int] = (512, 512),
    shape_types: Optional[List[str]] = None,
    random_seed: Optional[int] = None
) -> np.ndarray:
    """
    Generate a test dataset with multiple masks of different shapes.
    
    Parameters:
    -----------
    n_masks : int, default=100
        Number of masks to generate
    size : tuple, default=(512, 512)
        Size of each mask (height, width)
    shape_types : list of str, optional
        List of shape types to include. Options: 'circle', 'ellipse', 'protrusions', 'random'.
        If None, includes all types.
    random_seed : int, optional
        Random seed for reproducibility
        
    Returns:
    --------
    np.ndarray
        Array of binary masks with shape (n_masks, height, width)
        
    Example:
    --------
    >>> masks = create_test_dataset(n_masks=10, shape_types=['circle', 'ellipse'])
    >>> print(masks.shape)
    (10, 512, 512)
    """
    if random_seed is not None:
        np.random.seed(random_seed)
    
    # Default shape types
    if shape_types is None:
        shape_types = ['circle', 'ellipse', 'protrusions', 'random']
    
    # Create array to hold masks
    masks = np.zeros((n_masks, size[0], size[1]), dtype=np.uint8)
    
    logger.info(f"Generating {n_masks} test masks")
    
    for i in range(n_masks):
        # Select random shape type
        shape_type = np.random.choice(shape_types)
        
        # Random center (ensure it's not too close to the edge)
        padding = 100
        center_x = np.random.randint(padding, size[1] - padding)
        center_y = np.random.randint(padding, size[0] - padding)
        center = (center_x, center_y)
        
        if shape_type == 'circle':
            # Random radius
            radius = np.random.randint(30, 80)
            # Random noise
            noise = np.random.uniform(0, 0.3)
            masks[i] = create_circle_mask(size, center, radius, noise)
            
        elif shape_type == 'ellipse':
            # Random axes and angle
            major_axis = np.random.randint(50, 100)
            minor_axis = np.random.randint(20, major_axis - 10)
            angle = np.random.uniform(0, 180)
            noise = np.random.uniform(0, 0.3)
            masks[i] = create_ellipse_mask(size, center, (major_axis, minor_axis), angle, noise)
            
        elif shape_type == 'protrusions':
            # Random protrusion parameters
            radius = np.random.randint(30, 70)
            num_protrusions = np.random.randint(3, 8)
            protrusion_size = np.random.randint(5, 15)
            protrusion_distance = np.random.uniform(1.1, 1.3)
            masks[i] = create_shape_with_protrusions(
                size, center, radius, num_protrusions, protrusion_size, protrusion_distance
            )
            
        elif shape_type == 'random':
            # Random polygon
            min_vertices = np.random.randint(5, 10)
            max_vertices = min_vertices + np.random.randint(2, 8)
            min_radius = np.random.randint(30, 50)
            max_radius = min_radius + np.random.randint(20, 40)
            masks[i] = create_random_shape(
                size, center, max_vertices, min_vertices, (min_radius, max_radius)
            )
    
    logger.info(f"Test dataset generation complete")
    return masks


def save_test_dataset(
    output_path: str,
    n_masks: int = 100,
    size: Tuple[int, int] = (512, 512),
    shape_types: Optional[List[str]] = None,
    random_seed: Optional[int] = None
) -> str:
    """
    Generate and save a test dataset to a NPY file.
    
    Parameters:
    -----------
    output_path : str
        Path to save the NPY file
    n_masks : int, default=100
        Number of masks to generate
    size : tuple, default=(512, 512)
        Size of each mask (height, width)
    shape_types : list of str, optional
        List of shape types to include. Options: 'circle', 'ellipse', 'protrusions', 'random'.
        If None, includes all types.
    random_seed : int, optional
        Random seed for reproducibility
        
    Returns:
    --------
    str
        Path to the saved NPY file
        
    Example:
    --------
    >>> path = save_test_dataset('test_masks.npy', n_masks=10)
    >>> print(f"Saved to {path}")
    """
    # Generate dataset
    masks = create_test_dataset(n_masks, size, shape_types, random_seed)
    
    # Save to file
    logger.info(f"Saving test dataset to {output_path}")
    np.save(output_path, masks)
    
    return output_path
