---
title: "LTSA<T>"
description: "Local Tangent Space Alignment for nonlinear dimensionality reduction."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Preprocessing.DimensionalityReduction`

Local Tangent Space Alignment for nonlinear dimensionality reduction.

## For Beginners

LTSA is an improved version of LLE that:

- Uses tangent spaces (local linear approximations) at each point
- Better handles points with different local geometries
- More mathematically principled than standard LLE

Use cases:

- When standard LLE produces poor results
- Manifolds with varying curvature
- When you need more stable embeddings

## How It Works

LTSA is a manifold learning algorithm that uses local tangent spaces to compute
a global embedding. It estimates the tangent space at each point using PCA on
neighbors, then aligns these local coordinate systems.

The algorithm:

1. Find k-nearest neighbors for each point
2. Compute local tangent space via PCA on each neighborhood
3. Compute local coordinates in tangent space
4. Align tangent spaces by minimizing reconstruction error

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `LTSA(Int32,Int32,Nullable<Int32>,Int32[])` | Creates a new instance of `LTSA`. |

## Properties

| Property | Summary |
|:-----|:--------|
| `Embedding` | Gets the embedding result. |
| `NComponents` | Gets the number of components (dimensions). |
| `NNeighbors` | Gets the number of neighbors. |
| `SupportsInverseTransform` | Gets whether this transformer supports inverse transformation. |

## Methods

| Method | Summary |
|:-----|:--------|
| `FitCore(Matrix<>)` | Fits LTSA and computes the embedding. |
| `GetFeatureNamesOut(String[])` | Gets the output feature names after transformation. |
| `InverseTransformCore(Matrix<>)` | Inverse transformation is not supported. |
| `TransformCore(Matrix<>)` | Returns the embedding computed during Fit. |

