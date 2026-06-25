---
title: "EarthMoversDistance<T>"
description: "Earth Mover's Distance (EMD) / Wasserstein Distance for point cloud comparison."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Metrics`

Earth Mover's Distance (EMD) / Wasserstein Distance for point cloud comparison.

## How It Works

EMD measures the minimum cost to transform one distribution into another,
where cost is the sum of distances moved weighted by the amount moved.
Uses an approximation based on optimal assignment for efficiency.

Lower EMD indicates better point cloud similarity.

**Usage in 3D AI:**

- Point cloud generation evaluation
- 3D shape comparison
- More robust than Chamfer Distance for some applications

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `EarthMoversDistance(Int32,Double)` | Initializes a new instance of the EMD metric. |

## Methods

| Method | Summary |
|:-----|:--------|
| `Compute(Tensor<>,Tensor<>)` | Computes approximate EMD between two point clouds using Sinkhorn algorithm. |

## Fields

| Field | Summary |
|:-----|:--------|
| `_epsilon` | Regularization parameter for Sinkhorn algorithm. |
| `_iterations` | Number of iterations for the Sinkhorn algorithm approximation. |
| `_numOps` | The numeric operations provider for type T. |

