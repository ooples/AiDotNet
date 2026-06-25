---
title: "ChamferDistance<T>"
description: "Chamfer Distance metric for 3D point cloud comparison."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Metrics`

Chamfer Distance metric for 3D point cloud comparison.

## How It Works

Chamfer Distance measures the average squared distance from each point in one set
to its nearest neighbor in the other set, computed bidirectionally.
CD(X,Y) = (1/|X|)Σ_x min_y ||x-y||² + (1/|Y|)Σ_y min_x ||y-x||²

Lower Chamfer Distance indicates better point cloud similarity.

**Usage in 3D AI:**

- Point cloud completion evaluation
- 3D reconstruction quality
- Shape generation evaluation
- NeRF/Gaussian Splatting geometry quality

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `ChamferDistance(Boolean)` | Initializes a new instance of the Chamfer Distance metric. |

## Methods

| Method | Summary |
|:-----|:--------|
| `Compute(Tensor<>,Tensor<>)` | Computes Chamfer Distance between two point clouds. |
| `ComputeBatch(Tensor<>,Tensor<>)` | Computes Chamfer Distance for batched point clouds. |
| `ComputeOneWay(Tensor<>,Tensor<>)` | Computes one-way Chamfer Distance from source to target. |
| `ComputePointDistance(Tensor<>,Tensor<>,Int32,Int32,Int32)` | Computes distance between two specific points. |
| `ExtractPointCloud(Tensor<>,Int32,Int32,Int32)` | Extracts a single point cloud from a batch. |

## Fields

| Field | Summary |
|:-----|:--------|
| `_numOps` | The numeric operations provider for type T. |
| `_squared` | Whether to use squared distances (faster) or Euclidean distances. |

