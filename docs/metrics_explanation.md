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

### 3. Roundness Measures

**How they're calculated:**
We calculate roundness using several different approaches:

1. **Standard Roundness (`roundness`)**: Uses the standard isoperimetric ratio formula, which is the primary measure of roundness:
   ```python
   # Standard roundness formula using isoperimetric inequality
   roundness = 4 * np.pi * area / (perimeter**2)
   ```

2. **Alternative Roundness (`roundness_alt`)**: Maximum of the standard measure and the equivalent circle method - provides an optimistic assessment of circularity:
   ```python
   # Choose the better roundness measure (closer to 1.0 for a circle)
   roundness_alt = max(roundness, roundness_equivalent)
   ```

3. **Equivalent Circle Roundness (`roundness_equivalent`)**: Based on the ratio of equivalent circle perimeter to actual perimeter:
   ```python
   # Calculate diameter of a circle with the same area
   equivalent_diameter = np.sqrt(4 * area / np.pi)
   # Calculate perimeter of this equivalent circle
   equiv_circle_perimeter = np.pi * equivalent_diameter
   # Roundness based on perimeter comparison
   roundness_equivalent = equiv_circle_perimeter / perimeter
   ```

**Interpretation:**
- All three measures range from 0 to 1, where 1 represents a perfect circle.
- Lower values indicate more irregular or elongated shapes.
- The isoperimetric inequality guarantees that circles maximize the standard formula.
- The equivalent circle measure often handles discretization effects better.
- For most shapes, the two approaches will be very close, but for shapes with complex boundaries or discretization artifacts, they may differ.
- The alternative measure (`roundness_alt`) simply takes the maximum of the two, providing the most optimistic assessment of circularity.

### 4. Equivalent Diameter

**How it's calculated:**
The equivalent diameter is the diameter of a circle with the same area as the shape:

```python
equivalent_diameter = np.sqrt(4 * area / np.pi)
```

**Interpretation:**
- Represents the diameter of a circle that would have the same area as the shape.
- Useful for comparing sizes of differently shaped objects.
- For a circle, this equals the actual diameter.

## Ellipse Metrics

### 5. Ellipticity

**How it's calculated:**
Ellipticity is calculated by fitting an ellipse to the contour and taking the ratio of the minor axis to the major axis. This produces a value between 0 and 1, where 1 represents a perfect circle and values approach 0 for increasingly elongated shapes.

```python
ellipse = cv2.fitEllipse(contour)
center, axes, angle = ellipse

major_axis = max(axes)
minor_axis = min(axes)
ellipticity = minor_axis / major_axis
```

Values very close to 1.0 (within 0.01) are normalized to exactly 1.0.

**Interpretation:**
- A value of 1.0 indicates a circle (minor axis = major axis).
- Lower values indicate more elongated shapes.
- For example, an ellipticity of 0.5 means the shape is twice as long in one direction as the other.
- This measure is inverse to the traditional aspect ratio, focusing on circularity (values closer to 1 mean more circular).

### 6. Major and Minor Axes

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

### 7. Orientation

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

### 8. Solidity

**How it's calculated:**
Solidity is the ratio of the contour area to its convex hull area:

```python
hull = cv2.convexHull(contour)
contour_area = cv2.contourArea(contour)
hull_area = cv2.contourArea(hull)
solidity = contour_area / hull_area
```

**Interpretation:**
- Ranges from 0 to 1, where 1 means the shape is completely convex (has no indentations).
- Lower values indicate shapes with significant concavities or indentations.
- For example, a star shape would have lower solidity than a circle.

### 9. Convexity

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

### 10. Protrusion Count

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

### 11. Enhanced Protrusion Metrics

The following metrics provide more detailed analysis of protrusions using the improved morphological approach:

#### 11.1 Protrusion Mean Length

**How it's calculated:**
After isolating individual protrusions, the length of each protrusion is measured from its connection to the main body to its tip. The mean length is then calculated:

```python
# For each protrusion
length = distance_from_connection_to_tip

# Calculate mean across all protrusions
mean_length = np.mean([p['length'] for p in protrusions])
```

**Interpretation:**
- Measures the average extent or reach of protrusions from the main body.
- Larger values indicate longer, more pronounced protrusions.
- Useful for distinguishing between short, stubby protrusions and long, thin ones.

#### 11.2 Protrusion Mean Width

**How it's calculated:**
For each protrusion, the width is estimated as the shorter dimension of the minimum bounding rectangle. The mean width is calculated across all protrusions:

```python
# For each protrusion, calculate minimum width
rect = cv2.minAreaRect(protrusion_contour)
_, (width, height), _ = rect
width = min(width, height)

# Calculate mean across all protrusions
mean_width = np.mean([p['width'] for p in protrusions])
```

**Interpretation:**
- Measures the average thickness of protrusions.
- Helps distinguish between thin, finger-like protrusions and broad extensions.
- Combined with length, provides insight into the aspect ratio of protrusions.

#### 11.3 Protrusion Length CV (Coefficient of Variation)

**How it's calculated:**
The coefficient of variation (CV) of protrusion lengths is calculated as the standard deviation divided by the mean:

```python
# Calculate length variability
lengths = np.array([p['length'] for p in protrusions])
length_cv = np.std(lengths) / mean_length if mean_length > 0 else 0
```

**Interpretation:**
- Measures the uniformity of protrusion lengths.
- A value of 0 indicates all protrusions are exactly the same length.
- Higher values indicate greater variation in protrusion lengths.
- Useful for distinguishing between shapes with uniform protrusions and those with varied protrusion lengths.

#### 11.4 Protrusion Spacing Uniformity

**How it's calculated:**
This metric measures how evenly protrusions are spaced around the shape. It calculates the angular spacing between adjacent protrusions and evaluates their uniformity:

```python
# Sort protrusions by angle around the shape
angles = np.sort([p['angle'] for p in protrusions])

# Calculate angular differences (including wrap-around)
diffs = np.diff(angles)
diffs = np.append(diffs, 360 + angles[0] - angles[-1])

# Calculate coefficient of variation of the differences
angle_cv = np.std(diffs) / np.mean(diffs) if np.mean(diffs) > 0 else 0

# Convert to a uniformity score (0-1)
spacing_uniformity = 1 - min(angle_cv, 1.0)
```

**Interpretation:**
- Ranges from 0 to 1, where 1 represents perfectly uniform spacing.
- Lower values indicate protrusions clustered in certain regions.
- Higher values indicate protrusions that are evenly distributed around the shape.
- A value near 1.0 suggests a star-like shape with regular protrusions.

### 12. Skeleton Metrics

**How they're calculated:**
The skeleton is a 1-pixel-wide representation of the shape that preserves its topology, calculated using morphological thinning operations.

```python
from skimage.morphology import skeletonize
skeleton = skeletonize(mask > 0).astype(np.uint8)
```

From this skeleton, several metrics are derived:

#### 12.1 Skeleton Branches

**How it's calculated:**
Branch points are identified as skeleton pixels with more than 2 neighbors. Endpoints have exactly 1 neighbor. The number of branches is estimated based on graph theory principles.

```python
# Estimate number of branches
# Each branch has up to 2 endpoints, and branches share branch points
# Complex formula based on graph theory: E = V - C (edges = vertices - components)
# Assuming single component, branches = endpoints/2 + branch_points - 1
num_branches = max(0, num_endpoints/2 + num_branch_points - 1)
```

**Interpretation:**
- Counts the topological branches in the shape's skeleton.
- A simple circle or ellipse has 0 branches.
- Higher values indicate more complex, branching structures.
- Each protrusion typically contributes to the branch count.

#### 12.2 Skeleton Complexity

**How it's calculated:**
The skeleton complexity normalizes the number of branches by the shape's size (approximated by twice the radius).

```python
# Normalize by shape size (a circle's diameter)
complexity = num_branches / max(1, 2 * radius)
```

**Interpretation:**
- Measures branching complexity relative to the shape's size.
- Size-independent measure that allows comparison between shapes of different scales.
- Higher values indicate more intricate, branching structures per unit size.
- A simple circle or ellipse has complexity of 0.

#### 12.3 Skeleton Branch Length

**How it's calculated:**
The mean branch length is estimated by dividing the total number of skeleton pixels by the number of branches.

```python
# Calculate mean branch length
skeleton_pixels = np.sum(skeleton)
mean_branch_length = skeleton_pixels / max(1, num_branches)
```

**Interpretation:**
- Measures the average length of branches in the skeleton.
- Longer branches typically indicate more elongated protrusions.
- A shape with no branches will have this value undefined (reported as 0).

### 13. Fractal Dimension

**How it's calculated:**
The fractal dimension is calculated using the box-counting method, which measures how the detail of a pattern changes with scale.

```python
# Calculate box counts at different scales
box_sizes = [2, 4, 8, 16, 32, 64]
box_counts = [count_boxes(contour, size) for size in box_sizes]

# Log transform
log_box_sizes = np.log(box_sizes)
log_box_counts = np.log(box_counts)

# Linear regression to find the slope
slope = np.polyfit(log_box_sizes, log_box_counts, 1)[0]

# Fractal dimension is the negative of the slope
fractal_dimension = -slope
```

**Interpretation:**
- Ranges from 1.0 (smooth curves) to 2.0 (space-filling).
- A smooth circle has a fractal dimension of 1.0.
- Higher values indicate more complex, irregular boundaries.
- A very jagged, rough boundary approaches 2.0.
- Provides a measure of boundary complexity that complements roundness and convexity.

### 14. Boundary Entropy

**How it's calculated:**
Boundary entropy measures the unpredictability of curvature changes along the boundary by calculating the Shannon entropy of the curvature distribution.

```python
# Calculate curvature at each point
curvature = [calculate_turn_angle(point, prev, next) for point, prev, next in contour_triplets]

# Calculate entropy of the curvature distribution
histogram = np.histogram(curvature, bins=36, density=True)[0]
histogram = histogram[histogram > 0]  # Remove empty bins
entropy = -np.sum(histogram * np.log(histogram))
```

**Interpretation:**
- Higher values indicate more unpredictable, varied boundary curvature.
- Low values suggest more predictable, regular boundaries.
- A perfect circle has low boundary entropy (all curvature values are similar).
- A shape with many different types of curves and angles has high boundary entropy.
- Complements fractal dimension by focusing on the variability rather than complexity.

## Implementation Notes

### Discretization Effects

When working with digital images, shapes are represented on a discrete pixel grid. This discretization can affect metric calculations, especially for small objects or metrics sensitive to boundary precision (like roundness).

To mitigate these effects, we've implemented several strategies:

1. **Contour smoothing**: For perimeter calculations, we slightly smooth the contour to reduce jagged pixel edges.

2. **Multiple roundness calculations**: We provide several methods to calculate roundness, each with its strengths.

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

4. **Advanced morphological approach** has been added that:
   - Isolates the main "body" of the shape using erosion
   - Separates individual protrusions for detailed analysis
   - Provides rich metrics about each protrusion

## Validation and Testing

To ensure these metrics work correctly, we test them on synthetic shapes with known properties:

1. **Perfect circles**: Should have roundness = 1.0, ellipticity = 1.0, solidity = 1.0, protrusions = 0.

2. **Ellipses with specific axis ratios**: Should have predictable ellipticity values.

3. **Shapes with controlled numbers of protrusions**: Should detect approximately the right number of protrusions.

4. **Circles with varying noise levels**: Shows how metrics degrade with increasing boundary irregularity.

These tests help validate the accuracy and reliability of the metrics calculation.
