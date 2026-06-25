---
title: "KernelPCA<T>"
description: "Kernel Principal Component Analysis for non-linear dimensionality reduction."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Preprocessing.DimensionalityReduction`

Kernel Principal Component Analysis for non-linear dimensionality reduction.

## For Beginners

Regular PCA finds straight-line patterns.
Kernel PCA can find curved patterns by mathematically "bending" the data:

- RBF kernel: Good for data with clusters or blobs
- Polynomial: Good for polynomial relationships
- Linear: Same as regular PCA

Think of it as finding principal components in a transformed space.

## How It Works

Kernel PCA is an extension of PCA that uses a kernel function to map
data into a higher-dimensional feature space where non-linear relationships
become linear, then performs standard PCA in that space.

This allows capturing non-linear relationships that standard PCA cannot.
Common kernels include RBF (Gaussian), polynomial, and sigmoid.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `KernelPCA(Int32,KernelType,Double,Double,Double,Boolean,Double,Int32[])` | Creates a new instance of `KernelPCA`. |

## Properties

| Property | Summary |
|:-----|:--------|
| `Gamma` | Gets the gamma parameter for RBF and polynomial kernels. |
| `Kernel` | Gets the kernel type. |
| `Lambdas` | Gets the eigenvalues. |
| `NComponents` | Gets the number of components. |
| `SupportsInverseTransform` | Gets whether this transformer supports inverse transformation. |

## Methods

| Method | Summary |
|:-----|:--------|
| `FitCore(Matrix<>)` | Fits KernelPCA by computing the kernel matrix and its eigenvectors. |
| `GetFeatureNamesOut(String[])` | Gets the output feature names after transformation. |
| `InverseTransformCore(Matrix<>)` | Transforms data back to approximate original space. |
| `TransformCore(Matrix<>)` | Transforms the data using kernel PCA. |

