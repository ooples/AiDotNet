---
title: "TNetLayer<T>"
description: "Implements a Transformation Network (T-Net) for learning spatial transformations of point clouds."
section: "API Reference"
---

`Layers` · `AiDotNet.PointCloud.Layers`

Implements a Transformation Network (T-Net) for learning spatial transformations of point clouds.

## How It Works

**For Beginners:** T-Net learns to align and normalize point clouds before processing.

What T-Net does:

- Learns a transformation matrix to apply to input points
- Aligns point clouds to a canonical orientation
- Makes the network more robust to rotations and translations
- Helps the network focus on shape rather than orientation

How it works:

1. Takes point cloud as input
2. Processes it through small neural network
3. Outputs a transformation matrix (e.g., 3x3 for spatial, KxK for feature)
4. Applies this matrix to transform the input

Two types of T-Net in PointNet:

- Input T-Net: 3x3 matrix to align XYZ coordinates
- Feature T-Net: KxK matrix to align high-dimensional features

Benefits:

- Achieves invariance to rigid transformations
- Normalizes point cloud orientation
- Improves classification and segmentation accuracy

Example:

- Input: Point cloud that might be rotated randomly
- T-Net learns: "Rotate this cloud 45 degrees to align it"
- Output: Aligned point cloud in standard orientation

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `TNetLayer(Int32,Int32,Int32[],Int32[])` | Initializes a new instance of the TNetLayer class. |

