---
title: "MiniBatchSparsePCA<T>"
description: "Mini-batch Sparse PCA using online dictionary learning."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Preprocessing.DimensionalityReduction`

Mini-batch Sparse PCA using online dictionary learning.

## For Beginners

Think of this as SparsePCA on a budget:

- SparsePCA looks at ALL your data at once (memory intensive)
- MiniBatchSparsePCA looks at small pieces at a time (memory efficient)
- Results are similar, but mini-batch is faster for large datasets
- Trade-off: Slightly less accurate but much more scalable

## How It Works

MiniBatchSparsePCA is a faster, memory-efficient version of SparsePCA that
processes data in mini-batches instead of using the full dataset. This makes
it suitable for large datasets that don't fit in memory.

The algorithm uses online dictionary learning with mini-batches, updating
the components incrementally as it processes each batch.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `MiniBatchSparsePCA(Int32,Double,Double,Int32,Int32,Double,Boolean,Nullable<Int32>,Int32[])` | Creates a new instance of `MiniBatchSparsePCA`. |

## Properties

| Property | Summary |
|:-----|:--------|
| `Alpha` | Gets the sparsity regularization parameter. |
| `BatchSize` | Gets the batch size. |
| `Components` | Gets the sparse components (each row is a component). |
| `Mean` | Gets the mean of each feature. |
| `NComponents` | Gets the number of components. |
| `NSamplesSeen` | Gets the number of samples seen during fitting. |
| `Ridge` | Gets the ridge regularization parameter. |
| `SupportsInverseTransform` | Gets whether this transformer supports inverse transformation. |

## Methods

| Method | Summary |
|:-----|:--------|
| `FitCore(Matrix<>)` | Fits Mini-batch Sparse PCA using online dictionary learning. |
| `GetFeatureNamesOut(String[])` | Gets the output feature names after transformation. |
| `InverseTransformCore(Matrix<>)` | Transforms data back to original space. |
| `TransformCore(Matrix<>)` | Transforms the data by projecting onto sparse components. |

