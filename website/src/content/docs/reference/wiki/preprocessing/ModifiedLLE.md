---
title: "ModifiedLLE<T>"
description: "Modified Locally Linear Embedding with regularization."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Preprocessing.DimensionalityReduction`

Modified Locally Linear Embedding with regularization.

## For Beginners

Modified LLE is more stable than standard LLE:

- Works better when you have many neighbors
- Less sensitive to noise
- Produces more consistent results

Use cases:

- When standard LLE is unstable
- High number of neighbors relative to dimensions
- Noisy data

## How It Works

Modified LLE adds regularization to the standard LLE algorithm to improve
numerical stability when the number of neighbors exceeds the input dimensionality.

The algorithm:

1. Find k-nearest neighbors for each point
2. Compute reconstruction weights with regularization
3. Compute embedding by minimizing reconstruction error

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `ModifiedLLE(Int32,Int32,Double,Nullable<Int32>,Int32[])` | Creates a new instance of `ModifiedLLE`. |

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
| `FitCore(Matrix<>)` | Fits Modified LLE and computes the embedding. |
| `GetFeatureNamesOut(String[])` | Gets the output feature names after transformation. |
| `InverseTransformCore(Matrix<>)` | Inverse transformation is not supported. |
| `TransformCore(Matrix<>)` | Returns the embedding computed during Fit. |

