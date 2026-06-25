---
title: "LargeVis<T>"
description: "LargeVis for large-scale visualization and dimensionality reduction."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Preprocessing.DimensionalityReduction`

LargeVis for large-scale visualization and dimensionality reduction.

## For Beginners

LargeVis is efficient for large datasets because:

- Uses approximate nearest neighbors for scalability
- Negative sampling reduces computation vs full graph
- Asynchronous updates enable parallel processing
- Layout preserves local neighborhood relationships

Use cases:

- Datasets with millions of points
- When t-SNE is too slow
- Interactive visualization systems
- Document and image embedding visualization

## How It Works

LargeVis is designed for visualizing large-scale high-dimensional data. It builds a
k-NN graph, computes edge weights based on shared neighbors, and uses asynchronous
stochastic gradient descent with negative sampling for optimization.

The algorithm:

1. Construct approximate k-NN graph using random projection trees
2. Compute edge weights using shared neighbor similarity
3. Initialize embedding with random projection or PCA
4. Optimize using negative sampling SGD (similar to word2vec)

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `LargeVis(Int32,Int32,Int32,Int32,Double,Double,Nullable<Int32>,Int32[])` | Creates a new instance of `LargeVis`. |

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
| `FitCore(Matrix<>)` | Fits LargeVis and computes the embedding. |
| `GetFeatureNamesOut(String[])` | Gets the output feature names after transformation. |
| `InverseTransformCore(Matrix<>)` | Inverse transformation is not supported. |
| `TransformCore(Matrix<>)` | Returns the embedding computed during Fit. |

