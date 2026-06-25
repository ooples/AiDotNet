---
title: "SparsePCA<T>"
description: "Sparse Principal Component Analysis using L1 regularization."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Preprocessing.DimensionalityReduction`

Sparse Principal Component Analysis using L1 regularization.

## For Beginners

Sparse PCA creates "simpler" principal components:

- Standard PCA: Component = 0.3*Feature1 + 0.2*Feature2 + 0.1*Feature3 + ...
- Sparse PCA: Component = 0.5*Feature1 + 0*Feature2 + 0.4*Feature3 + 0*...
- Zeros make it easier to interpret what each component represents

## How It Works

SparsePCA finds sparse principal components by applying L1 regularization
(LASSO-like penalty) to the component loadings. This results in components
where many loadings are exactly zero, making them more interpretable.

Unlike standard PCA where each component is a combination of ALL features,
sparse PCA produces components that depend on only a subset of features.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `SparsePCA(Int32,Double,Double,Int32,Double,Nullable<Int32>,Int32[])` | Creates a new instance of `SparsePCA`. |

## Properties

| Property | Summary |
|:-----|:--------|
| `Alpha` | Gets the sparsity regularization parameter. |
| `Components` | Gets the sparse components (each row is a component). |
| `Error` | Gets the reconstruction error for each iteration. |
| `Mean` | Gets the mean of each feature. |
| `NComponents` | Gets the number of components. |
| `Ridge` | Gets the ridge regularization parameter. |
| `SupportsInverseTransform` | Gets whether this transformer supports inverse transformation. |

## Methods

| Method | Summary |
|:-----|:--------|
| `FitCore(Matrix<>)` | Fits Sparse PCA using coordinate descent with L1 regularization. |
| `GetFeatureNamesOut(String[])` | Gets the output feature names after transformation. |
| `InverseTransformCore(Matrix<>)` | Transforms data back to original space. |
| `TransformCore(Matrix<>)` | Transforms the data by projecting onto sparse components. |

