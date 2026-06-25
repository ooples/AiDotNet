---
title: "IPointCloudClassification<T>"
description: "Defines functionality for point cloud classification tasks."
section: "API Reference"
---

`Interfaces` · `AiDotNet.PointCloud.Interfaces`

Defines functionality for point cloud classification tasks.

## How It Works

**For Beginners:** Point cloud classification determines what object or category an entire point cloud represents.

Think of classification as recognizing what an object is:

- Input: A complete point cloud of an object
- Output: The category the object belongs to
- It's like looking at a 3D scan and saying "this is a chair" or "this is a table"

Common classification benchmarks:

- ModelNet40: 40 categories of 3D objects (chair, table, car, airplane, etc.)
- ShapeNet: Large-scale dataset with many object categories
- ScanNet: Real-world scanned objects and scenes

Applications:

- Object recognition in 3D scans
- Quality control in manufacturing (identify defective parts)
- Archaeological artifact classification
- Medical imaging (classify anatomical structures)

## Methods

| Method | Summary |
|:-----|:--------|
| `ClassifyPointCloud(Tensor<>)` | Classifies a point cloud into one of the predefined categories. |

