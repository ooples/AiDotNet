---
title: "RandomProjection<T>"
description: "Random Projection for dimensionality reduction."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Preprocessing.DimensionalityReduction`

Random Projection for dimensionality reduction.

## For Beginners

Random projection is surprisingly effective:

- It's very fast (just matrix multiplication)
- It preserves distances approximately (guaranteed by math!)
- Great for preprocessing before other algorithms
- Works well for very high-dimensional data

Use cases:

- Speeding up distance-based algorithms (kNN, clustering)
- Reducing memory for large datasets
- Preprocessing before other dimensionality reduction

## How It Works

Random projection reduces dimensionality by projecting data onto a random subspace.
Despite its simplicity, it has strong theoretical guarantees via the Johnson-Lindenstrauss lemma:
pairwise distances are approximately preserved with high probability.

Two projection types are supported:

- Gaussian: Random matrix with entries from N(0, 1/k)
- Sparse: Random matrix with mostly zeros (faster, memory efficient)

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `RandomProjection(Nullable<Int32>,Nullable<Double>,RandomProjectionType,Double,Nullable<Int32>,Int32[])` | Creates a new instance of `RandomProjection`. |

## Properties

| Property | Summary |
|:-----|:--------|
| `NComponents` | Gets the number of components. |
| `ProjectionMatrix` | Gets the projection matrix. |
| `ProjectionType` | Gets the projection type. |
| `SupportsInverseTransform` | Gets whether this transformer supports inverse transformation. |

## Methods

| Method | Summary |
|:-----|:--------|
| `FitCore(Matrix<>)` | Fits the random projection by generating the projection matrix. |
| `GetFeatureNamesOut(String[])` | Gets the output feature names after transformation. |
| `InverseTransformCore(Matrix<>)` | Inverse transformation is not supported. |
| `TransformCore(Matrix<>)` | Transforms data by projecting onto the random subspace. |

