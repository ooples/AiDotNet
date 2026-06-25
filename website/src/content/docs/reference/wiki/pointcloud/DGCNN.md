---
title: "DGCNN<T>"
description: "Implements Dynamic Graph CNN (DGCNN) for point cloud processing."
section: "API Reference"
---

`Models & Types` · `AiDotNet.PointCloud.Models`

Implements Dynamic Graph CNN (DGCNN) for point cloud processing.

## For Beginners

DGCNN treats point clouds as graphs and uses edge convolutions to learn features.

## How It Works

Key innovations of DGCNN:

- Dynamic graph construction: Rebuilds neighborhood graph at each layer based on learned features
- Edge convolution: Learns features from edges connecting nearby points
- Captures local geometric structure more effectively than PointNet
- Adapts to the feature space, not just spatial coordinates

How DGCNN differs from PointNet:

- PointNet: Processes each point independently, then aggregates
- DGCNN: Explicitly models relationships between neighboring points
- Dynamic graphs: Neighborhoods change as features evolve through layers

Edge Convolution explained:

1. For each point, find K nearest neighbors (in feature space or spatial)
2. Compute edge features: combine point feature with neighbor features
3. Apply MLP to edge features
4. Aggregate (max pool) edge features for each point
5. Result: New features that incorporate local structure

Why dynamic graphs are powerful:

- Early layers: Use spatial proximity (XYZ coordinates)
- Later layers: Use semantic similarity (learned features)
- Example: Points on same chair leg become neighbors even if spatially distant

Architecture:

1. Multiple EdgeConv layers with increasing feature dimensions
2. Each layer rebuilds k-NN graph based on current features
3. Concatenate features from all EdgeConv layers
4. Max pooling for global features
5. Fully connected layers for classification/segmentation

Applications:

- Classification: Achieves state-of-the-art on ModelNet40
- Part segmentation: Excellent for identifying object parts
- Semantic segmentation: Captures fine-grained geometric details
- Better than PointNet at capturing local structure

Example - chair classification:

- Layer 1: Find spatial neighbors (nearby points)
- Layer 2: Find points with similar low-level features (edges, corners)
- Layer 3: Find points with similar mid-level features (vertical bars, flat surfaces)
- Layer 4: Find points with similar high-level features (legs, back, seat)
- Final: Combine all levels to recognize "chair"

Reference: "Dynamic Graph CNN for Learning on Point Clouds"
by Wang et al., ACM Transactions on Graphics 2019

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `DGCNN` | Initializes a new instance of the DGCNN class with default options. |
| `DGCNN(DGCNNOptions,ILossFunction<>,DGCNNOptions)` | Initializes a new instance of the DGCNN class with configurable options. |
| `DGCNN(Int32,Int32,Int32[],Boolean,Double,ILossFunction<>)` | Initializes a new instance of the DGCNN class. |

## Methods

| Method | Summary |
|:-----|:--------|
| `GetOptions` |  |

