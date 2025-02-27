"""
seg_ana: A package for characterizing morphological features of organoid segmentations.

This package provides tools for loading, processing, and analyzing segmentation masks
to extract morphological features of organoids.
"""

__version__ = "0.1.0"

# Import core functionality
from .core.loader import load_and_process, get_batch_info
from .core.metrics import analyze_batch, calculate_all_metrics
