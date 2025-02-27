# Testing with Synthetic Data

This guide explains how to use the synthetic data generation features in `seg-ana` to validate your metric calculations. By creating shapes with known theoretical properties, you can verify that the metrics calculations are working correctly.

## Overview

The `synthetic.py` module allows you to create:
- Perfect circles
- Ellipses with specific axis ratios
- Shapes with controlled numbers of protrusions
- Random polygon shapes
- Noisy variations of all the above

By comparing the calculated metrics on these shapes to their theoretical values, you can validate your analysis pipeline.

## Shape Types and Expected Metrics

### Perfect Circles

A perfect circle has these theoretical properties:
- Roundness: 1.0 (exactly)
- Ellipticity: 1.0 (major axis = minor axis)
- Solidity: 1.0 (no concavities)

```python
# Using standard metrics
from seg_ana.core.synthetic import create_circle_mask
from seg_ana.core.metrics import calculate_all_metrics

# Create a perfect circle
circle_mask = create_circle_mask(size=(256, 256), radius=50, noise=0.0)

# Calculate metrics
metrics = calculate_all_metrics(circle_mask)
print(f"Circle roundness: {metrics['roundness']}")  # Should be very close to 1.0
print(f"Circle ellipticity: {metrics['ellipticity']}")  # Should be very close to 1.0
print(f"Circle solidity: {metrics['solidity']}")  # Should be very close to 1.0
```

### Using Improved Metrics

For more accurate roundness calculations, use the improved metrics module:

```python
# Using improved metrics for better roundness calculation
from seg_ana.core.synthetic import create_circle_mask
from seg_ana.core.metrics_improved import calculate_all_metrics, create_mathematical_circle

# Option 1: Create a circle using OpenCV and measure with improved metrics
opencv_circle = create_circle_mask(size=(256, 256), radius=50, noise=0.0)
metrics_opencv = calculate_all_metrics(opencv_circle)
print(f"OpenCV circle roundness: {metrics_opencv['roundness']}")  # Should be 1.0

# Option 2: Create a mathematically perfect circle for even better results
math_circle = create_mathematical_circle(size=(256, 256), radius=50)
metrics_math = calculate_all_metrics(math_circle)
print(f"Mathematical circle roundness: {metrics_math['roundness']}")  # Should be exactly 1.0
```

### Comparing Standard vs. Improved Metrics

To comprehensively compare the standard and improved metric calculations:

```bash
# Run the metrics comparison test
python test_improved_metrics.py
```

This will generate a comparison of both approaches for circles with different radii and ellipses with different axis ratios, saving the results in the `improved_metrics_test` directory.

### Ellipses with Known Axis Ratios

Ellipses have these theoretical properties:
- Ellipticity: exactly the ratio of major axis to minor axis
- Roundness: decreases as ellipticity increases

```python
from seg_ana.core.synthetic import create_ellipse_mask
from seg_ana.core.metrics_improved import calculate_all_metrics

# Create an ellipse with 2:1 axis ratio
ellipse_mask = create_ellipse_mask(size=(256, 256), axes=(60, 30))

# Calculate metrics
metrics = calculate_all_metrics(ellipse_mask)
print(f"Ellipse ellipticity: {metrics['ellipticity']}")  # Should be close to 2.0
print(f"Ellipse roundness: {metrics['roundness']}")  # Should be < 1.0
```

### Shapes with Protrusions

Shapes with protrusions have these properties:
- Solidity: decreases with more/larger protrusions
- Roundness: decreases with more/larger protrusions
- Protrusions count: should detect the number of protrusions (may not exactly match the creation parameter)

```python
from seg_ana.core.synthetic import create_shape_with_protrusions
from seg_ana.core.metrics_improved import calculate_all_metrics

# Create a shape with exactly 5 protrusions
shape_mask = create_shape_with_protrusions(
    size=(256, 256), 
    radius=50,
    num_protrusions=5, 
    protrusion_size=10,
    protrusion_distance=1.3
)

# Calculate metrics
metrics = calculate_all_metrics(shape_mask)
print(f"Detected protrusions: {metrics['protrusions']}")
print(f"Solidity: {metrics['solidity']}")  # Lower for shapes with more protrusions
print(f"Roundness: {metrics['roundness']}")  # Lower for shapes with protrusions
```

## Visualizing and Exporting Masks

### Saving Mask Visualizations

To visualize and export a variety of masks with their metrics:

```bash
# Generate and save a collection of masks
python save_masks.py
```

This script creates a directory called `mask_exports` containing:
- Circle masks created with different methods
- Ellipses with different axis ratios
- Shapes with varying numbers of protrusions
- Circles with different noise levels
- A summary visualization of all shapes

### Validating Circles Specifically

To focus on validating circle metrics and investigating the roundness calculation:

```bash
# Validate circle metrics and visualize results
python validate_circles.py
```

This creates a directory called `validation_output` with detailed visualizations and comparison of different circle creation methods.

### Exporting Individual Masks

You can export individual masks for closer inspection:

```python
from seg_ana.core.validation import export_mask_visualization, save_mask_to_file
from seg_ana.core.synthetic import create_circle_mask

# Create a mask
circle_mask = create_circle_mask(size=(256, 256), radius=50)

# Save visualization with metrics
export_mask_visualization(circle_mask, "circle_viz.png")

# Save raw mask data (NPY) and image (PNG)
npy_path, png_path = save_mask_to_file(circle_mask, "circle_mask")
```

### Detailed Contour Analysis

For an in-depth analysis of contour properties:

```python
from seg_ana.core.validation import export_mask_with_analyzed_contour
from seg_ana.core.synthetic import create_circle_mask

# Create a mask
circle_mask = create_circle_mask(size=(256, 256), radius=50)

# Export detailed contour analysis
export_mask_with_analyzed_contour(circle_mask, "circle_analysis.png")
```

## Validation Workflow

### 1. Create Shapes with Known Properties

```python
import numpy as np
from seg_ana.core.synthetic import create_circle_mask, create_ellipse_mask
from seg_ana.core.metrics_improved import calculate_all_metrics, create_mathematical_circle

# Create shapes with known properties
circle_opencv = create_circle_mask(size=(256, 256), radius=50)
circle_math = create_mathematical_circle(size=(256, 256), radius=50)
ellipse_2_1 = create_ellipse_mask(size=(256, 256), axes=(60, 30))  # 2:1 ratio
ellipse_3_1 = create_ellipse_mask(size=(256, 256), axes=(90, 30))  # 3:1 ratio

# Save for validation
np.save('test_circle_opencv.npy', circle_opencv)
np.save('test_circle_math.npy', circle_math)
np.save('test_ellipse_2_1.npy', ellipse_2_1)
np.save('test_ellipse_3_1.npy', ellipse_3_1)
```

### 2. Run Analysis on These Shapes

```bash
poetry run seg-ana analyze test_circle_opencv.npy --output opencv_metrics.csv
poetry run seg-ana analyze test_circle_math.npy --output math_metrics.csv
poetry run seg-ana analyze test_ellipse_2_1.npy --output ellipse_2_1_metrics.csv
poetry run seg-ana analyze test_ellipse_3_1.npy --output ellipse_3_1_metrics.csv
```

### 3. Verify Metrics Match Expected Values

Open the CSV files and check:
- Circle should have roundness close to 1.0
- 2:1 ellipse should have ellipticity close to 2.0
- 3:1 ellipse should have ellipticity close to 3.0

## Command-line Generation

The `generate` command provides a convenient way to create test data:

```bash
# Generate multiple shapes of different types
poetry run seg-ana generate test_shapes.npy --num 10 --shapes circle,ellipse

# Generate only ellipses with a specific random seed for reproducibility
poetry run seg-ana generate test_ellipses.npy --num 5 --shapes ellipse --seed 42
```

## Systematic Testing

For comprehensive validation, test each metric systematically with shapes of varying properties:

### Testing Ellipticity Calculation

```python
import pandas as pd
import numpy as np
from seg_ana.core.synthetic import create_ellipse_mask
from seg_ana.core.metrics_improved import calculate_all_metrics

# Test ellipticity calculation with increasing axis ratios
results = []
for ratio in range(1, 6):
    major_axis = 60
    minor_axis = int(major_axis / ratio)
    mask = create_ellipse_mask(axes=(major_axis, minor_axis))
    metrics = calculate_all_metrics(mask)
    results.append({
        'expected_ratio': ratio,
        'measured_ellipticity': metrics['ellipticity']
    })

# Convert to DataFrame and save
df = pd.DataFrame(results)
print(df)
```

### Testing Roundness with Noise

```python
# Test how noise affects roundness of a circle
noise_levels = [0, 0.1, 0.2, 0.3, 0.4, 0.5]
results = []

for noise in noise_levels:
    mask = create_circle_mask(radius=50, noise=noise)
    metrics = calculate_all_metrics(mask)
    results.append({
        'noise_level': noise,
        'roundness': metrics['roundness'],
        'solidity': metrics['solidity']
    })

df = pd.DataFrame(results)
print(df)
```

### Testing Protrusion Detection

```python
# Test protrusion detection with increasing numbers of protrusions
results = []
for num_protrusions in range(0, 11, 2):  # 0, 2, 4, 6, 8, 10
    mask = create_shape_with_protrusions(
        radius=50,
        num_protrusions=num_protrusions,
        protrusion_size=10,
        protrusion_distance=1.3
    )
    metrics = calculate_all_metrics(mask)
    results.append({
        'created_protrusions': num_protrusions,
        'detected_protrusions': metrics['protrusions'],
        'solidity': metrics['solidity']
    })

df = pd.DataFrame(results)
print(df)
```

## Tips for Effective Validation

1. **Use mathematical circles**: For perfect circle tests, use `create_mathematical_circle()` from the improved metrics module for the most accurate results.

2. **Use improved metrics**: The `metrics_improved` module provides more accurate roundness calculations, especially for perfect shapes.

3. **Use random_seed for reproducibility**: When validating, always set a random seed to ensure you get the same shapes each time.

4. **Test edge cases**: Generate shapes with extreme properties (very elongated ellipses, shapes with many protrusions) to ensure the metrics handle these correctly.

5. **Create noisy shapes**: Use the `noise` parameter in shape creation to test how robust your metrics are to irregular boundaries.

6. **Compare related metrics**: Check relationships between metrics (e.g., as solidity decreases, roundness often decreases too).

7. **Visualize the shapes**: It can be helpful to save and view the shapes alongside their metrics to verify your understanding.

8. **Batch test multiple parameters**: Create systematic variations to understand how each parameter affects the metrics.

9. **Document expected value ranges**: Based on your tests, document the expected ranges for each metric for your specific use case.

By using synthetic data with known properties, you can validate your metrics calculations and gain confidence in your analysis of real organoid segmentations.
