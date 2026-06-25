---
title: "LaplacianEigenmaps<T>"
description: "Laplacian Eigenmaps for nonlinear dimensionality reduction."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Preprocessing.DimensionalityReduction`

Laplacian Eigenmaps for nonlinear dimensionality reduction.

## For Beginners

Laplacian Eigenmaps finds a low-dimensional representation where:

- Connected points in the graph stay close together
- The embedding respects the local geometry of the data
- It's similar to spectral clustering but for dimensionality reduction

Use cases:

- Manifold learning when data lies on a curved surface
- Image segmentation and clustering
- When you want to preserve local connectivity

## How It Works

Laplacian Eigenmaps constructs a weighted graph from the data and finds a low-dimensional
embedding that preserves local neighborhood relationships by minimizing a cost function
based on the graph Laplacian.

The algorithm:

1. Constructs a k-nearest neighbor or epsilon-neighborhood graph
2. Computes edge weights using a kernel (e.g., heat kernel)
3. Computes the graph Laplacian: L = D - W
4. Finds eigenvectors of the generalized eigenvalue problem: L*y = λ*D*y

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `LaplacianEigenmaps(Int32,Int32,Nullable<Double>,LaplacianAffinityType,Double,Nullable<Int32>,Int32[])` | Creates a new instance of `LaplacianEigenmaps`. |

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
| `FitCore(Matrix<>)` | Fits Laplacian Eigenmaps and computes the embedding. |
| `GetFeatureNamesOut(String[])` | Gets the output feature names after transformation. |
| `InverseTransformCore(Matrix<>)` | Inverse transformation is not supported. |
| `TransformCore(Matrix<>)` | Returns the embedding computed during Fit. |

