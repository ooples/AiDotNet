---
title: "TriMAP<T>"
description: "TriMAP: Large-scale Dimensionality Reduction Using Triplets."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Preprocessing.DimensionalityReduction`

TriMAP: Large-scale Dimensionality Reduction Using Triplets.

## For Beginners

TriMAP preserves distance relationships using triplets:

- A triplet (i, j, k) means point i is closer to j than to k
- Optimization ensures these relationships hold in the embedding
- More accurate than t-SNE/UMAP for many datasets
- Less sensitive to parameter tuning

Use cases:

- Large-scale visualization (millions of points)
- When t-SNE/UMAP produces poor results
- When you need faithful global structure
- Complex datasets with hierarchical structure

## How It Works

TriMAP is a dimensionality reduction method that uses triplet constraints to preserve
both local and global structure. It outperforms t-SNE and UMAP on many datasets while
being more robust to hyperparameter choices.

The algorithm:

1. Generate triplets (anchor, positive, negative) based on distance relationships
2. Initialize embedding using PCA
3. Optimize using triplet loss to preserve distance rankings
4. Weight triplets to balance local and global structure

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `TriMAP(Int32,Int32,Int32,Int32,Int32,Double,Nullable<Int32>,Int32[])` | Creates a new instance of `TriMAP`. |

## Properties

| Property | Summary |
|:-----|:--------|
| `Embedding` | Gets the embedding result. |
| `NComponents` | Gets the number of components (dimensions). |
| `SupportsInverseTransform` | Gets whether this transformer supports inverse transformation. |

## Methods

| Method | Summary |
|:-----|:--------|
| `FitCore(Matrix<>)` | Fits TriMAP and computes the embedding. |
| `GetFeatureNamesOut(String[])` | Gets the output feature names after transformation. |
| `InverseTransformCore(Matrix<>)` | Inverse transformation is not supported. |
| `TransformCore(Matrix<>)` | Returns the embedding computed during Fit. |

