---
title: "MaxPoolingLayer<T>"
description: "Implements global max pooling for point clouds to extract global features."
section: "API Reference"
---

`Layers` · `AiDotNet.PointCloud.Layers`

Implements global max pooling for point clouds to extract global features.

## How It Works

**For Beginners:** Max pooling takes the maximum value across all points for each feature channel.

How it works:

- Input: N points, each with C features [N, C]
- Operation: For each feature channel, find the maximum value across all N points
- Output: A single vector of C features [1, C]

Why it's useful:

- Creates a global representation of the entire point cloud
- Achieves permutation invariance (order of points doesn't matter)
- Reduces dimensionality from many points to one feature vector

Example:

- Input: 1024 points with 64 features each = [1024, 64]
- Max pooling across points
- Output: 1 global feature vector with 64 features = [1, 64]

This is a key component in PointNet for making the network invariant to point order.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `MaxPoolingLayer(Int32)` | Initializes a new instance of the MaxPoolingLayer class. |

