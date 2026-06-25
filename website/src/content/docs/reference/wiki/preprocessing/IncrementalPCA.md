---
title: "IncrementalPCA<T>"
description: "Incremental Principal Component Analysis for large datasets."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Preprocessing.DimensionalityReduction`

Incremental Principal Component Analysis for large datasets.

## For Beginners

Regular PCA needs all data in memory at once.
Incremental PCA processes data in chunks:

- Feed data in batches (e.g., 1000 rows at a time)
- Updates its understanding of the data with each batch
- Produces similar principal components as regular PCA
- Uses much less memory for large datasets

## How It Works

IncrementalPCA processes data in batches, making it suitable for datasets
too large to fit in memory. It produces similar results to standard PCA
but with lower memory requirements.

The algorithm updates the covariance matrix incrementally as each batch
is processed, then computes principal components from the final estimate.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `IncrementalPCA(Int32,Int32,Boolean,Int32[])` | Creates a new instance of `IncrementalPCA`. |

## Properties

| Property | Summary |
|:-----|:--------|
| `BatchSize` | Gets the batch size for incremental updates. |
| `Components` | Gets the principal components. |
| `ExplainedVariance` | Gets the explained variance for each component. |
| `ExplainedVarianceRatio` | Gets the explained variance ratio for each component. |
| `Mean` | Gets the mean of each feature. |
| `NComponents` | Gets the number of components to keep. |
| `NSamplesSeen` | Gets the number of samples seen during fitting. |
| `SupportsInverseTransform` | Gets whether this transformer supports inverse transformation. |
| `Whiten` | Gets whether whitening is applied. |

## Methods

| Method | Summary |
|:-----|:--------|
| `FitCore(Matrix<>)` | Fits IncrementalPCA by processing data in batches. |
| `GetFeatureNamesOut(String[])` | Gets the output feature names after transformation. |
| `InverseTransformCore(Matrix<>)` | Transforms data back to original space. |
| `PartialFit(Matrix<>)` | Partially fits the model with a new batch of data. |
| `TransformCore(Matrix<>)` | Transforms the data by projecting onto principal components. |

