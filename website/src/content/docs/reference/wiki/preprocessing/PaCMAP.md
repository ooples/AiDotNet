---
title: "PaCMAP<T>"
description: "PaCMAP: Pairwise Controlled Manifold Approximation."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Preprocessing.DimensionalityReduction`

PaCMAP: Pairwise Controlled Manifold Approximation.

## For Beginners

PaCMAP improves on UMAP/t-SNE by:

- Preserving global structure better through mid-near and further pairs
- Using controlled pair selection instead of random sampling
- Dynamically adjusting focus from global to local structure
- Being more robust to hyperparameter choices

Use cases:

- When you need faithful global structure
- When t-SNE/UMAP produces fragmented clusters
- When relative distances between clusters matter
- Biological data, image embeddings, document visualization

## How It Works

PaCMAP is a dimensionality reduction method designed to preserve both local and global
structure. It uses three types of point pairs with carefully controlled weights during
optimization to achieve better structure preservation than t-SNE or UMAP.

The algorithm:

1. Create three types of pairs: nearby (local), mid-near (intermediate), further (global)
2. Initialize embedding using PCA
3. Optimize using attractive/repulsive forces with dynamic weighting
4. Gradually shift from global to local focus during optimization

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `PaCMAP(Int32,Int32,Double,Double,Int32,Double,Nullable<Int32>,Int32[])` | Creates a new instance of `PaCMAP`. |

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
| `FitCore(Matrix<>)` | Fits PaCMAP and computes the embedding. |
| `GetFeatureNamesOut(String[])` | Gets the output feature names after transformation. |
| `InverseTransformCore(Matrix<>)` | Inverse transformation is not supported. |
| `TransformCore(Matrix<>)` | Returns the embedding computed during Fit. |

