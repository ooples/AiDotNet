---
title: "HessianLLE<T>"
description: "Hessian Locally Linear Embedding for nonlinear dimensionality reduction."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Preprocessing.DimensionalityReduction`

Hessian Locally Linear Embedding for nonlinear dimensionality reduction.

## For Beginners

Hessian LLE improves on LLE by:

- Using curvature information (second derivatives)
- Producing more faithful global embeddings
- Better handling of manifolds with varying curvature

Use cases:

- When standard LLE produces distorted embeddings
- Manifolds with non-uniform curvature
- When you need more accurate distance preservation

## How It Works

Hessian LLE is an improvement over standard LLE that estimates the Hessian
(second derivative) of the embedding function. It uses a quadratic form to
measure local curvature and produces more globally coherent embeddings.

The algorithm:

1. Find k-nearest neighbors for each point
2. Compute local Hessian estimator using quadratic polynomials
3. Build global Hessian matrix
4. Find embedding by minimizing Hessian-based cost function

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `HessianLLE(Int32,Int32,Nullable<Int32>,Int32[])` | Creates a new instance of `HessianLLE`. |

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
| `FitCore(Matrix<>)` | Fits Hessian LLE and computes the embedding. |
| `GetFeatureNamesOut(String[])` | Gets the output feature names after transformation. |
| `InverseTransformCore(Matrix<>)` | Inverse transformation is not supported. |
| `TransformCore(Matrix<>)` | Returns the embedding computed during Fit. |

