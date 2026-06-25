---
title: "PointNetPlusPlus<T>"
description: "Implements the PointNet++ architecture for hierarchical point cloud processing."
section: "API Reference"
---

`Models & Types` · `AiDotNet.PointCloud.Models`

Implements the PointNet++ architecture for hierarchical point cloud processing.

## For Beginners

PointNet++ extends PointNet by adding hierarchical feature learning at multiple scales.

## How It Works

Key improvements over PointNet:

- Hierarchical structure: Processes point clouds at multiple resolutions
- Local context: Captures fine-grained local patterns
- Multi-scale grouping: Learns features at different scales simultaneously
- Better generalization: More robust to non-uniform point density

Architecture components:

1. Set Abstraction Layers: Hierarchically group points and extract features
- Sampling: Select subset of points as centroids
- Grouping: Find neighboring points around each centroid
- PointNet layer: Extract features from each local region
2. Feature Propagation Layers: Upsample features for segmentation tasks
- Interpolation: Propagate features from coarse to fine levels
- Skip connections: Combine with features from encoder

Why hierarchical learning matters:

- Different patterns exist at different scales (like edges vs. shapes in images)
- Local context provides detailed geometry information
- Global context provides overall shape understanding
- Combining both gives comprehensive understanding

Applications:

- Fine-grained classification
- Part segmentation (identifying specific parts of objects)
- Semantic segmentation (labeling each point)
- Better performance on complex, detailed shapes

Example use case - autonomous driving:

- Coarse level: Identify general object shapes (car, pedestrian)
- Medium level: Recognize object parts (wheels, windows)
- Fine level: Detect details (door handles, mirrors)

Reference: "PointNet++: Deep Hierarchical Feature Learning on Point Sets in a Metric Space"
by Qi et al., NeurIPS 2017

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `PointNetPlusPlus` | Initializes a new instance of the PointNetPlusPlus class with default options. |
| `PointNetPlusPlus(Int32,Int32[],Double[],Int32[][],Boolean,ILossFunction<>)` | Initializes a new instance of the PointNetPlusPlus class. |
| `PointNetPlusPlus(PointNetPlusPlusOptions,ILossFunction<>,PointNetPlusPlusOptions)` | Initializes a new instance of the PointNetPlusPlus class with configurable options. |

## Methods

| Method | Summary |
|:-----|:--------|
| `GetOptions` |  |

