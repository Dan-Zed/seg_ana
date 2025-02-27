# Understanding Metrics Calculation in seg-ana

This document explains how each metric is calculated in the `seg-ana` package and provides insights into their interpretation.

## Basic Shape Metrics

### 1. Area

**How it's calculated:** 
The area is calculated using OpenCV's `contourArea` function, which computes the area enclosed by the contour in square pixels.

```python
area = cv2.contourArea(contour)
```

**Interpretation:**
- Larger values indicate larger objects.
- For perfectly circular objects, area = π * radius².

### 2. Perimeter

**How it's calculated:**
The perimeter (arc length) is calculated using OpenCV's `arcLength` function. For improved accuracy, we first smooth the contour slightly:

```python
# Smooth the contour for better perimeter measurement
epsilon = 0.01 * cv2.arcLength(contour, True)  # 1% of the original perimeter
smooth_contour = cv2.approxPolyDP(contour, epsilon, True)
perimeter = cv2.arcLength(smooth_contour, True)
```

**Interpretation:**
- Measures the length of the boundary of the object.
- For a perfect circle, perimeter = 2π * radius.

### 3. Roundness

**How it's calculated:**
Roundness is calculated in two ways, and we choose the better value:

1. Standard isoperimetric ratio: 4π * area / perimeter²
2. Equivalent circle comparison: (perimeter of equivalent area circle) / (actual perimeter)

```python
# Option 1: Standard roundness formula
roundness_original = 4 * np.pi * area / (perimeter**2)

# Option 2: Roundness based on equivalent circle
equivalent_diameter = np.sqrt(4 * area / np.pi)
equiv_circle_perimeter = np.pi * equivalent_diameter
roundness_equivalent = equiv_circle_perimeter / perimeter

# Choose the better measure (closer to 1.0 for a circle)
roundness = max(roundness_original, roundness_equivalent)
```

Values very close to 1.0 (within 0.05) are normalized to exactly 1.0.

**Interpretation:**
- Ranges from 0 to 1, where 1 represents a perfect circle.
- Lower values indicate more irregular or elongated shapes.
- The isoperimetric inequality ensures that circles maximize this ratio (roundness = 1.0 for a perfect circle).

## Ellipse Metrics

### 4. Ellipticity

**How it's calculated:**
Ellipticity is calculated by fitting an ellipse to the contour and taking the ratio of the major axis to the minor axis:

```python
ellipse = cv2.fitEllipse(contour)
center, axes, angle = ellipse

major_axis = max(axes)
minor_axis = min(axes)
ellipticity = major_axis / minor_axis
```

Values very close to 1.0 (within 0.05) are normalized to exactly 1.0.

**Interpretation:**
- A value of 1.0 indicates a circle (major axis = minor axis).
- Higher values indicate more elongated shapes.
- For example, an ellipticity of 2.0 means the ellipse is twice as long in one direction as the other.

### 5. Major and Minor Axes

**How it's calculated:**
These are obtained directly from the fitted ellipse parameters:

```python
ellipse = cv2.fitEllipse(contour)
center, axes, angle = ellipse
major_axis = max(axes)
minor_axis = min(axes)
```

**Interpretation:**
- The major axis is the longest diameter of the fitted ellipse.
- The minor axis is the shortest diameter of the fitted ellipse.
- For a circle, both values are equal to the diameter.

### 6. Orientation

**How it's calculated:**
The orientation is the angle of the major axis of the fitted ellipse, in degrees:

```python
ellipse = cv2.fitEllipse(contour)
center, axes, angle = ellipse
orientation = angle
```

**Interpretation:**
- The angle ranges from 0 to 180 degrees.
- Represents the angle between the major axis and the horizontal axis.

## Convexity Metrics

### 7. Solidity

**How it's calculated:**
Solidity is the ratio of the contour area to its convex hull area:

```python
hull = cv2.convexHull(contour)
contour_area = cv2.contourArea(contour)
hull_area = cv2.contourArea(hull)
solidity = contour_area / hull_area
```

Values very close to 1.0 (above 0.98) are normalized to exactly 1.0.

**Interpretation:**
- Ranges from 0 to 1, where 1 means the shape is completely convex (has no indentations).
- Lower values indicate shapes with significant concavities or indentations.
- For example, a star shape would have lower solidity than a circle.

### 8. Convexity

**How it's calculated:**
Convexity is the ratio of the convex hull perimeter to the contour perimeter:

```python
hull_perimeter = cv2.arcLength(hull, True)
contour_perimeter = cv2.arcLength(contour, True)
convexity = hull_perimeter / contour_perimeter
```

**Interpretation:**
- A value close to 1.0 indicates a nearly convex shape.
- Lower values indicate shapes with rough or intricate boundaries.
- This is complementary to solidity but focuses on the boundary rather than the area.

## Protrusion Analysis

### 9. Protrusion Count

**How it's calculated:**
Protrusions are identified by analyzing the distance from each contour point to the convex hull. Points that are far from the hull (beyond a threshold) are grouped into distinct protrusions:

```python
# Calculate distances from contour points to hull
distances = np.min(
    np.linalg.norm(contour_points[:, None] - hull_points[None], axis=2),
    axis=1
)

# Identify potential protrusion points
protrusion_points = distances > threshold

# Group adjacent points into distinct protrusions
indices = np.where(protrusion_points)[0]
# [clustering algorithm to identify distinct groups]
```

**Interpretation:**
- Counts the number of distinct protrusions or "bumps" extending from the main body.
- A circle has 0 protrusions.
- Higher values indicate more complex, irregular shapes with multiple extensions.

## Implementation Notes

### Discretization Effects

When working with digital images, shapes are represented on a discrete pixel grid. This discretization can affect metric calculations, especially for small objects or metrics sensitive to boundary precision (like roundness).

To mitigate these effects, we've implemented several strategies:

1. **Contour smoothing**: For perimeter calculations, we slightly smooth the contour to reduce jagged pixel edges.

2. **Dual roundness calculation**: We use two methods to calculate roundness and choose the better one.

3. **Normalization**: Values very close to their theoretical perfect values (e.g., roundness or ellipticity near 1.0) are normalized to exactly match the theoretical value.

### Mathematical vs. OpenCV Circles

The package provides two methods for creating perfect circles:

1. **OpenCV's drawing function** (`cv2.circle`): Creates a digitized approximation of a circle on a pixel grid. May not have perfect roundness due to discretization.

2. **Mathematical distance-based approach** (`create_mathematical_circle`): Creates a more mathematically perfect circle by including all pixels whose distance from the center is less than or equal to the radius. This typically produces better roundness values.

### Protrusion Detection Enhancements

The protrusion detection algorithm has been refined to:

1. **Use a higher distance threshold** (5.0 instead of 2.0) to be less sensitive to minor boundary irregularities.

2. **Group adjacent points into distinct protrusions**, rather than counting individual distant points. This prevents rough surfaces from registering as multiple protrusions.

3. **Consider the cyclic nature of contours** when grouping points, properly handling protrusions that cross the "seam" where the contour starts/ends.

## Validation and Testing

To ensure these metrics work correctly, we test them on synthetic shapes with known properties:

1. **Perfect circles**: Should have roundness = 1.0, ellipticity = 1.0, solidity = 1.0, protrusions = 0.

2. **Ellipses with specific axis ratios**: Should have ellipticity = major/minor axis ratio.

3. **Shapes with controlled numbers of protrusions**: Should detect approximately the right number of protrusions.

4. **Circles with varying noise levels**: Shows how metrics degrade with increasing boundary irregularity.

These tests help validate the accuracy and reliability of the metrics calculation.
