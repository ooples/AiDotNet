---
title: "LocallyLinearEmbedding<T>"
description: "Locally Linear Embedding for nonlinear dimensionality reduction."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Preprocessing.DimensionalityReduction`

Locally Linear Embedding for nonlinear dimensionality reduction.

## For Beginners

LLE preserves local relationships:

- Each point is described by its neighbors
- The weights describe "how much" each neighbor contributes
- The embedding keeps these relationships intact
- Good for unfolding curved manifolds (like the Swiss roll)

## How It Works

LLE preserves local neighborhood structure by representing each point
as a weighted linear combination of its neighbors. The embedding is found
by preserving these reconstruction weights in lower dimensions.

The algorithm:

1. Find k nearest neighbors for each point
2. Compute reconstruction weights that best reconstruct each point from neighbors
3. Find low-dimensional embedding that preserves these weights

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `LocallyLinearEmbedding(Int32,Int32,Double,LLEMethod,Int32[])` | Creates a new instance of `LocallyLinearEmbedding`. |

## Properties

| Property | Summary |
|:-----|:--------|
| `Embedding` | Gets the embedding result. |
| `Method` | Gets the LLE method. |
| `NComponents` | Gets the number of components. |
| `NNeighbors` | Gets the number of neighbors. |
| `Regularization` | Gets the regularization parameter. |
| `SupportsInverseTransform` | Gets whether this transformer supports inverse transformation. |

## Methods

| Method | Summary |
|:-----|:--------|
| `FitCore(Matrix<>)` | Fits LLE by computing reconstruction weights and embedding. |
| `GetFeatureNamesOut(String[])` | Gets the output feature names after transformation. |
| `InverseTransformCore(Matrix<>)` | Inverse transformation is not supported. |
| `TransformCore(Matrix<>)` | Returns the embedding computed during Fit. |

