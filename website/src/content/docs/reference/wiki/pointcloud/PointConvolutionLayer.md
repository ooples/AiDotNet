---
title: "PointConvolutionLayer<T>"
description: "Implements a convolution layer specifically designed for point cloud data."
section: "API Reference"
---

`Layers` · `AiDotNet.PointCloud.Layers`

Implements a convolution layer specifically designed for point cloud data.

## How It Works

**For Beginners:** Unlike regular convolutions for images, point cloud convolutions work on unordered 3D points.

Key differences from image convolutions:

- Images have regular grid structure (pixels in rows/columns)
- Point clouds are unordered sets of 3D coordinates
- Must be invariant to point order (permutation invariant)
- Must handle varying number of points

This layer learns features from point neighborhoods by:

- Finding nearby points (local neighborhood)
- Aggregating features from these neighbors
- Learning weights that work regardless of point order

Applications:

- Feature extraction from local 3D geometry
- Learning shape patterns in point clouds
- Building blocks for PointNet-style architectures

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `PointConvolutionLayer(Int32,Int32,IActivationFunction<>)` | Initializes a new instance of the PointConvolutionLayer class. |

## Methods

| Method | Summary |
|:-----|:--------|
| `InitializeWeights(Int32,Int32)` | Initializes weights using He initialization for better convergence. |

