---
title: "IoU3D<T>"
description: "3D Intersection over Union (3D IoU) for voxel and bounding box evaluation."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Metrics`

3D Intersection over Union (3D IoU) for voxel and bounding box evaluation.

## How It Works

3D IoU measures the overlap between two 3D volumes.
IoU = Volume(Intersection) / Volume(Union)

**Usage in 3D AI:**

- Voxel-based 3D detection
- 3D bounding box evaluation
- Occupancy grid comparison

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `IoU3D` | Initializes a new instance of the 3D IoU metric. |

## Methods

| Method | Summary |
|:-----|:--------|
| `ComputeBoxIoU([],[])` | Computes 3D IoU between two axis-aligned bounding boxes. |
| `ComputeVoxelIoU(Tensor<>,Tensor<>)` | Computes 3D IoU between two voxel grids (binary occupancy). |

## Fields

| Field | Summary |
|:-----|:--------|
| `_numOps` | The numeric operations provider for type T. |

