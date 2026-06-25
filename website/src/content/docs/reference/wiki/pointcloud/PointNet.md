---
title: "PointNet<T>"
description: "Implements the PointNet architecture for processing point cloud data."
section: "API Reference"
---

`Models & Types` · `AiDotNet.PointCloud.Models`

Implements the PointNet architecture for processing point cloud data.

## For Beginners

PointNet is a pioneering deep learning architecture designed to directly process point clouds.

## How It Works

Key innovations of PointNet:

- Directly processes unordered point sets (no need to convert to voxels or images)
- Permutation invariant: output doesn't change if you shuffle the input points
- Learns both local and global features
- Uses spatial transformer networks (T-Net) for alignment

Architecture overview:

1. Input transformation: T-Net learns to align input points
2. Multi-layer perceptron (MLP): Processes each point independently
3. Feature transformation: Another T-Net aligns learned features
4. More MLPs: Further feature extraction
5. Max pooling: Aggregates information from all points
6. Global feature vector: Represents the entire point cloud
7. Classification/Segmentation: Task-specific layers

Why it's important:

- First successful deep learning approach for raw point clouds
- Achieves state-of-the-art results on ModelNet40 classification
- Foundation for many subsequent point cloud methods
- Widely used in robotics, autonomous driving, and 3D vision

Reference: "PointNet: Deep Learning on Point Sets for 3D Classification and Segmentation"
by Qi et al., CVPR 2017

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `PointNet` | Initializes a new instance of the PointNet class with default options. |
| `PointNet(Int32,Boolean,Boolean,ILossFunction<>)` | Initializes a new instance of the PointNet class. |
| `PointNet(PointNetOptions,ILossFunction<>,PointNetOptions)` | Initializes a new instance of the PointNet class with configurable options. |

## Methods

| Method | Summary |
|:-----|:--------|
| `GetOptions` |  |

