---
title: "RandomizedPCA<T>"
description: "Randomized PCA using randomized SVD for efficient computation."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Preprocessing.DimensionalityReduction`

Randomized PCA using randomized SVD for efficient computation.

## For Beginners

Randomized PCA is faster because:

- It only computes the components you need (not all)
- Random projection preserves structure efficiently
- Power iteration improves accuracy if needed
- Works well for low-rank data

Use cases:

- Very large datasets where standard PCA is slow
- When you only need top few components
- Streaming or online scenarios
- Memory-constrained environments

## How It Works

Randomized PCA uses randomized algorithms to efficiently compute principal components
without computing the full SVD. It's much faster than standard PCA for large datasets
while providing accurate approximations of the top components.

The algorithm:

1. Generate random projection matrix
2. Form sample matrix Y = A * Ω (project data to random subspace)
3. Orthonormalize Y using QR decomposition
4. Form B = Q^T * A (project to Q's range)
5. Compute SVD of B to get principal components

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `RandomizedPCA(Int32,Int32,Int32,Nullable<Int32>,Int32[])` | Creates a new instance of `RandomizedPCA`. |

## Properties

| Property | Summary |
|:-----|:--------|
| `Components` | Gets the principal components (each row is a component). |
| `ExplainedVariance` | Gets the explained variance for each component. |
| `NComponents` | Gets the number of components (dimensions). |
| `SingularValues` | Gets the singular values. |
| `SupportsInverseTransform` | Gets whether this transformer supports inverse transformation. |

## Methods

| Method | Summary |
|:-----|:--------|
| `FitCore(Matrix<>)` | Fits Randomized PCA using randomized SVD. |
| `GetExplainedVarianceRatio` | Gets the total explained variance ratio (proportion of total variance explained by selected components). |
| `GetFeatureNamesOut(String[])` | Gets the output feature names after transformation. |
| `InverseTransformCore(Matrix<>)` | Reconstructs data from the reduced representation. |
| `TransformCore(Matrix<>)` | Transforms data by projecting onto principal components. |

