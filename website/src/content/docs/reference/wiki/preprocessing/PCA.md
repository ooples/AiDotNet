---
title: "PCA<T>"
description: "Principal Component Analysis for dimensionality reduction."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Preprocessing.DimensionalityReduction`

Principal Component Analysis for dimensionality reduction.

## For Beginners

PCA transforms your features into new features called
"principal components" that are:

- Uncorrelated with each other
- Ordered by importance (how much variance they explain)
- Linear combinations of your original features

Example: 100 features might be reduced to 10 principal components that capture
95% of the information in your data.

## How It Works

PCA finds the directions of maximum variance in the data and projects the data
onto these principal components. This reduces dimensionality while preserving
as much variance as possible.

PCA is useful for:

- Reducing the number of features while retaining most information
- Removing multicollinearity between features
- Visualizing high-dimensional data
- Noise reduction

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `PCA(Nullable<Int32>,Nullable<Double>,Boolean,Int32[])` | Creates a new instance of `PCA`. |

## Properties

| Property | Summary |
|:-----|:--------|
| `Components` | Gets the principal components (each row is a component). |
| `ExplainedVariance` | Gets the explained variance for each component. |
| `ExplainedVarianceRatio` | Gets the explained variance ratio for each component. |
| `Mean` | Gets the mean of each feature. |
| `NComponents` | Gets the number of components to keep. |
| `NComponentsOut` | Gets the number of components after fitting. |
| `SingularValues` | Gets the singular values. |
| `SupportsInverseTransform` | Gets whether this transformer supports inverse transformation. |
| `VarianceRatio` | Gets the target variance ratio to retain. |
| `Whiten` | Gets whether whitening is applied. |

## Methods

| Method | Summary |
|:-----|:--------|
| `FitCore(Matrix<>)` | Fits PCA by computing principal components. |
| `GetFeatureNamesOut(String[])` | Gets the output feature names after transformation. |
| `InverseTransformCore(Matrix<>)` | Transforms data back to original space. |
| `TransformCore(Matrix<>)` | Transforms the data by projecting onto principal components. |

