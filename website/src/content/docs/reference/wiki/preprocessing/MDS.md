---
title: "MDS<T>"
description: "Multidimensional Scaling for dimensionality reduction."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Preprocessing.DimensionalityReduction`

Multidimensional Scaling for dimensionality reduction.

## For Beginners

MDS tries to place points in 2D/3D such that:

- Points that were close in high-D stay close in low-D
- Points that were far apart stay far apart
- Unlike t-SNE/UMAP, MDS tries to preserve actual distances, not just neighborhoods

Classical MDS: Preserves exact distances (works well when data is linear)
Non-metric MDS: Preserves distance rankings (more flexible, works better for complex data)

## How It Works

MDS finds a low-dimensional embedding that preserves pairwise distances between points.
Classical MDS uses eigendecomposition of the double-centered distance matrix.
Non-metric MDS uses iterative optimization to preserve distance rankings.

MDS is useful for:

- Visualizing similarity/dissimilarity data
- Preserving pairwise relationships
- When you have a distance matrix rather than feature vectors

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `MDS(Int32,MDSType,MDSMetric,Int32,Double,Boolean,Nullable<Int32>,Int32[])` | Creates a new instance of `MDS`. |

## Properties

| Property | Summary |
|:-----|:--------|
| `Embedding` | Gets the embedding result. |
| `MdsType` | Gets the MDS type (classical or non-metric). |
| `NComponents` | Gets the number of components (dimensions). |
| `Stress` | Gets the final stress value (goodness of fit). |
| `SupportsInverseTransform` | Gets whether this transformer supports inverse transformation. |

## Methods

| Method | Summary |
|:-----|:--------|
| `FitCore(Matrix<>)` | Fits MDS and computes the embedding. |
| `GetFeatureNamesOut(String[])` | Gets the output feature names after transformation. |
| `InverseTransformCore(Matrix<>)` | Inverse transformation is not supported. |
| `TransformCore(Matrix<>)` | Returns the embedding computed during Fit. |

