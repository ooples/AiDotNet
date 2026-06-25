---
title: "MinMaxScaler<T>"
description: "Scales features to a given range, typically [0, 1]."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Preprocessing.Scalers`

Scales features to a given range, typically [0, 1].

## For Beginners

This scaler squishes all your data into a specific range:

- The smallest value becomes the minimum of the range (default 0)
- The largest value becomes the maximum of the range (default 1)
- Everything else is proportionally scaled in between

This is useful when:

- Your algorithm requires data in a specific range
- You want to preserve the relationships between values
- You don't want outliers to heavily influence the scaling

## How It Works

Min-max scaling transforms features by scaling each feature to a given range.
The default range is [0, 1]. The transformation is: X_scaled = (X - X_min) / (X_max - X_min) * (max - min) + min

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `MinMaxScaler(Double,Double,Int32[])` | Creates a new instance of `MinMaxScaler` with a custom range. |
| `MinMaxScaler(Int32[])` | Creates a new instance of `MinMaxScaler` with default range [0, 1]. |

## Properties

| Property | Summary |
|:-----|:--------|
| `DataMax` | Gets the maximum value of each feature computed during fitting. |
| `DataMin` | Gets the minimum value of each feature computed during fitting. |
| `FeatureRangeMax` | Gets the maximum value of the target feature range. |
| `FeatureRangeMin` | Gets the minimum value of the target feature range. |
| `SupportsInverseTransform` | Gets whether this transformer supports inverse transformation. |

## Methods

| Method | Summary |
|:-----|:--------|
| `FitCore(Matrix<>)` | Computes the minimum and maximum of each feature from the training data. |
| `GetFeatureNamesOut(String[])` | Gets the output feature names after transformation. |
| `InverseTransformCore(Matrix<>)` | Reverses the min-max scaling transformation. |
| `TransformCore(Matrix<>)` | Transforms the data by applying min-max scaling. |

