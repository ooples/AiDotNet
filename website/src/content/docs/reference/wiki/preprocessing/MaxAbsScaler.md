---
title: "MaxAbsScaler<T>"
description: "Scales each feature by its maximum absolute value."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Preprocessing.Scalers`

Scales each feature by its maximum absolute value.

## For Beginners

This scaler divides each feature by its largest absolute value:

- The largest value (positive or negative) becomes 1 or -1
- All other values are proportionally scaled
- Zero values remain zero (preserves sparsity)

This is useful when:

- You have sparse data and want to preserve zeros
- You don't want to center your data
- You want values bounded between -1 and 1

## How It Works

Max absolute scaling transforms each feature by dividing by its maximum absolute value,
resulting in features with values in the range [-1, 1]. This scaler does not shift or center
the data, so it preserves sparsity.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `MaxAbsScaler(Int32[])` | Creates a new instance of `MaxAbsScaler`. |

## Properties

| Property | Summary |
|:-----|:--------|
| `MaxAbsolute` | Gets the maximum absolute value of each feature computed during fitting. |
| `SupportsInverseTransform` | Gets whether this transformer supports inverse transformation. |

## Methods

| Method | Summary |
|:-----|:--------|
| `FitCore(Matrix<>)` | Computes the maximum absolute value of each feature from the training data. |
| `GetFeatureNamesOut(String[])` | Gets the output feature names after transformation. |
| `InverseTransformCore(Matrix<>)` | Reverses the max absolute scaling transformation. |
| `TransformCore(Matrix<>)` | Transforms the data by applying max absolute scaling. |

