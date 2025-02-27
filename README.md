# Seg-Ana: Segmentation Analysis Tool

A Python package for analyzing morphological features of organoid segmentations. It provides fast, memory-optimized processing of binary segmentation masks for extracting quantitative metrics.

## Features

- Fast, memory-optimized processing of segmentation masks
- Extraction of morphological metrics:
  - Area, perimeter, roundness
  - Ellipticity and orientation
  - Convexity and solidity 
  - Advanced protrusion detection and analysis
- Batch processing with multi-core support
- Detailed visualizations for metric calculation
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
poetry run seg-ana generate test_masks.npy --num 50 --size 512
```

For help with any command:

```bash
poetry run seg-ana <command> --help
```

### Batch Processing

```bash
# Process all .npy files in a directory
python scripts/batch_analyze.py /path/to/mask/directory --output_dir ./results

# Process with specific number of worker processes
python scripts/batch_analyze.py /path/to/mask/directory --workers 4
```

This will:
- Process all .npy files in the input directory
- Calculate metrics for each mask
- Generate detailed visualizations for each mask
- Create a CSV with all metrics
- Generate summary visualizations of the entire dataset

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

# Enhanced protrusion analysis
from seg_ana.core.protrusion_analysis import analyze_all_protrusions
protrusion_results = analyze_all_protrusions(
    masks[0], visualize=True, output_dir="./protrusion_analysis"
)
```

### Generating Test Data

To create synthetic data for testing or demonstration:

```python
from seg_ana.core.synthetic import create_test_dataset, save_test_dataset

# Generate and save a dataset with 50 masks of different shapes
save_test_dataset('test_masks.npy', n_masks=50, size=(512, 512))

# Create shapes with specific properties
from seg_ana.core.synthetic import create_shape_with_protrusions
shape = create_shape_with_protrusions(
    size=(512, 512), 
    radius=100, 
    num_protrusions=6,
    protrusion_size=25
)
```

## Examples

Check out the `examples/` directory for sample scripts:

```bash
# Run basic analysis example
python examples/basic_analysis.py

# Run batch processing example
python examples/batch_processing.py
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

- **Protrusion metrics**:
  - `protrusions`: Count of protrusions/extensions
  - `protrusion_mean_length`: Average length of protrusions
  - `protrusion_mean_width`: Average width of protrusions
  - `protrusion_length_cv`: Coefficient of variation of protrusion lengths
  - `protrusion_spacing_uniformity`: Measure of angular spacing uniformity (0-1)

## Visualization Tools

The package includes several visualization tools in the `scripts` directory:

```bash
# Visualize metrics calculations
python scripts/visualize_metrics.py

# Test and visualize protrusion detection
python scripts/test_protrusion_detection.py

# Detailed protrusion analysis visualization
python scripts/visualize_protrusions.py

# Compare isolated protrusion detection with standard method
python scripts/test_isolated_protrusions.py
```

## Performance Notes

This package is optimized for:
- In-memory processing of segmentation masks
- Efficiency on standard hardware (8-16 GB RAM)
- Vectorized operations using NumPy and OpenCV
- Parallel processing of multiple masks

## Development

### Testing

Run tests:

```bash
poetry run pytest
```

### Project Structure

```
seg_ana/
├── examples/                 # Example usage scripts
│   ├── basic_analysis.py     # Simple analysis example
│   └── batch_processing.py   # Batch processing example
├── outputs/                  # Output directory for tests and visualizations
├── scripts/                  # Utility scripts
│   ├── batch_analyze.py      # Batch processing script
│   ├── visualize_metrics.py  # Metrics visualization
│   ├── test_protrusion_detection.py # Protrusion detection testing
│   └── visualize_protrusions.py     # Protrusion analysis visualization
├── src/
│   └── seg_ana/
│       ├── core/
│       │   ├── loader.py            # Loading and preprocessing masks
│       │   ├── metrics.py           # Basic metrics calculation
│       │   ├── metrics_improved.py  # Enhanced metrics calculation
│       │   ├── protrusion_analysis.py # Advanced protrusion analysis
│       │   └── synthetic.py         # Test data generation
│       └── cli/
│           └── commands.py          # Command-line interface
├── tests/
│   ├── core/
│   └── cli/
└── docs/
    ├── tech_doc.md                  # Technical documentation
    ├── metrics_explanation.md       # Detailed metrics explanation
    └── progress_report_*.md         # Progress reports
```

## License

MIT
