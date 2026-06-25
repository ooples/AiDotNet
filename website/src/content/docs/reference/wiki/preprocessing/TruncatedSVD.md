---
title: "TruncatedSVD<T>"
description: "Truncated Singular Value Decomposition for dimensionality reduction."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Preprocessing.DimensionalityReduction`

Truncated Singular Value Decomposition for dimensionality reduction.

## For Beginners

SVD is similar to PCA but:

- Doesn't center the data (preserves sparsity in sparse matrices)
- Can be more memory-efficient for large sparse datasets
- Often used for text analysis (finding hidden topics in documents)

Example: In text analysis, TruncatedSVD can find that "car" and "automobile"
are related even if they never appear together in the same document.

## How It Works

TruncatedSVD performs dimensionality reduction by computing the truncated
singular value decomposition. Unlike PCA, it does not center the data,
making it suitable for sparse matrices (e.g., TF-IDF from text).

Also known as Latent Semantic Analysis (LSA) when applied to document-term matrices.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `TruncatedSVD(Int32,Int32,Int32,Int32[])` | Creates a new instance of `TruncatedSVD`. |

## Properties

| Property | Summary |
|:-----|:--------|
| `Components` | Gets the components (right singular vectors). |
| `ExplainedVariance` | Gets the explained variance for each component. |
| `ExplainedVarianceRatio` | Gets the explained variance ratio for each component. |
| `NComponents` | Gets the number of components. |
| `NIterations` | Gets the number of iterations for randomized SVD. |
| `SingularValues` | Gets the singular values. |
| `SupportsInverseTransform` | Gets whether this transformer supports inverse transformation. |

## Methods

| Method | Summary |
|:-----|:--------|
| `FitCore(Matrix<>)` | Fits TruncatedSVD by computing singular value decomposition. |
| `GetFeatureNamesOut(String[])` | Gets the output feature names after transformation. |
| `InverseTransformCore(Matrix<>)` | Transforms data back to original space. |
| `TransformCore(Matrix<>)` | Transforms the data by projecting onto singular vectors. |

