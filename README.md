# Seg-Ana: Segmentation Analysis Tool

A Python package for analyzing morphological features of organoid segmentations. It provides fast, memory-optimized processing of binary segmentation masks for extracting quantitative metrics.

## Features

- Fast, memory-optimized processing of segmentation masks
- Extraction of morphological metrics:
  - Area, perimeter, roundness
  - Ellipticity and orientation
  - Convexity and solidity 
  - Protrusion detection
- Command-line interface for easy usage
- Cross-platform (macOS, Linux)

## Installation

```bash
# Clone the repository
git clone https://github.com/your-username/seg-ana.git
cd seg-ana

# Install with Poetry (recommended)
poetry install
```

## Usage

### Command Line Interface

The package provides a command-line interface for common tasks:

```bash
# Analyze masks and export metrics to CSV
poetry run seg-ana analyze path/to/masks.npy --output results.csv

# Get information about masks
poetry run seg-ana info path/to/masks.npy

# Generate synthetic test data
poetry run seg-ana generate test_masks.npy --num 50 --size 256
```

For help with any command:

```bash
poetry run seg-ana <command> --help
```

### Python API

```python
import numpy as np
from seg_ana import load_and_process, analyze_batch

# Load and preprocess masks
masks = load_and_process('path/to/masks.npy', min_area=100)

# Calculate metrics for all masks
metrics_list = analyze_batch(masks)

# Access metrics for individual masks
first_mask_metrics = metrics_list[0]
print(f"Area: {first_mask_metrics['area']}")
print(f"Roundness: {first_mask_metrics['roundness']}")
```

### Generating Test Data

To create synthetic data for testing or demonstration:

```python
from seg_ana.core.synthetic import create_test_dataset, save_test_dataset

# Generate and save a dataset with 50 masks of different shapes
save_test_dataset('test_masks.npy', n_masks=50)

# Or create directly in memory
masks = create_test_dataset(n_masks=20, shape_types=['circle', 'ellipse'])
```

## Available Metrics

- **Basic metrics**: 
  - `area`: Surface area in pixels
  - `perimeter`: Perimeter length in pixels
  - `roundness`: Circularity measure (4π × area / perimeter²), ranges from 0-1
  - `equivalent_diameter`: Diameter of circle with equivalent area

- **Ellipse metrics**:
  - `ellipticity`: Ratio of major to minor axis
  - `major_axis`: Length of major axis
  - `minor_axis`: Length of minor axis
  - `orientation`: Angle of orientation in degrees

- **Convexity metrics**:
  - `solidity`: Ratio of contour area to convex hull area
  - `convexity`: Ratio of convex hull perimeter to contour perimeter
  - `convex_hull_area`: Area of the convex hull

- **Other**:
  - `protrusions`: Count of protrusions/extensions

## Performance Notes

This package is optimized for:
- In-memory processing of segmentation masks
- Efficiency on standard hardware (8-16 GB RAM)
- Vectorized operations using NumPy and OpenCV

## Development

### Testing

Run tests:

```bash
poetry run pytest
```

### Project Structure

```
seg_ana/
├── src/
│   └── seg_ana/
│       ├── core/
│       │   ├── loader.py      # Loading and preprocessing masks
│       │   ├── metrics.py     # Morphological metrics calculation
│       │   └── synthetic.py   # Test data generation
│       └── cli/
│           └── commands.py    # Command-line interface
├── tests/
│   ├── core/
│   └── cli/
└── docs/
    └── tech_doc.md           # Technical documentation
```

## License

MIT
