---
title: "SpectralEmbedding<T>"
description: "Spectral Embedding for nonlinear dimensionality reduction."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Preprocessing.DimensionalityReduction`

Spectral Embedding for nonlinear dimensionality reduction.

## For Beginners

Spectral Embedding uses graph theory:

- Build a graph where similar points are connected
- Use the graph's structure to find good coordinates
- Similar to what's used in spectral clustering
- Good for data with cluster structure

## How It Works

Spectral Embedding forms an affinity matrix from the data and computes
the eigenvectors of the graph Laplacian. This provides a low-dimensional
representation that preserves local connectivity.

The algorithm constructs a similarity graph and uses spectral decomposition
of the Laplacian matrix to find coordinates that respect graph structure.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `SpectralEmbedding(Int32,SpectralAffinity,Nullable<Double>,Int32,Nullable<Int32>,Int32[])` | Creates a new instance of `SpectralEmbedding`. |

## Properties

| Property | Summary |
|:-----|:--------|
| `Affinity` | Gets the affinity type. |
| `AffinityMatrix` | Gets the affinity matrix. |
| `Embedding` | Gets the embedding result. |
| `Gamma` | Gets the gamma parameter for RBF kernel. |
| `NComponents` | Gets the number of components. |
| `NNeighbors` | Gets the number of neighbors for nearest neighbors affinity. |
| `SupportsInverseTransform` | Gets whether this transformer supports inverse transformation. |

## Methods

| Method | Summary |
|:-----|:--------|
| `FitCore(Matrix<>)` | Fits Spectral Embedding by computing the graph Laplacian eigenvectors. |
| `GetFeatureNamesOut(String[])` | Gets the output feature names after transformation. |
| `InverseTransformCore(Matrix<>)` | Inverse transformation is not supported. |
| `TransformCore(Matrix<>)` | Returns the embedding computed during Fit. |

