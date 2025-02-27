"""
Core functionality for segmentation mask analysis.
"""
from .loader import (
    load_npy_file,
    preprocess_masks,
    load_and_process,
    get_batch_info
)

from .metrics import (
    get_largest_contour,
    calculate_basic_metrics,
    calculate_ellipse_metrics,
    calculate_convexity_metrics,
    calculate_all_metrics,
    analyze_batch,
    count_protrusions
)

from .synthetic import (
    create_circle_mask,
    create_ellipse_mask,
    create_shape_with_protrusions,
    create_random_shape,
    create_test_dataset,
    save_test_dataset
)
