---
title: "GlobalContrastScaler<T>"
description: "Scales features by adjusting contrast based on mean and standard deviation."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Preprocessing.Scalers`

Scales features by adjusting contrast based on mean and standard deviation.

## For Beginners

This scaler improves the "contrast" of your data:

- It centers values around 0.5 (the new average)
- Values above average become > 0.5, below average become < 0.5
- Most values end up between 0 and 1

Example: [68, 70, 71, 69, 72] → [0.3, 0.5, 0.6, 0.4, 0.7]
Now the differences between values are more visible.

## How It Works

Global contrast scaling transforms data using the formula: (x - mean) / (2 * stdDev) + 0.5
This centers values around 0.5 and typically results in values between 0 and 1.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `GlobalContrastScaler(Int32[])` | Creates a new instance of `GlobalContrastScaler`. |

## Properties

| Property | Summary |
|:-----|:--------|
| `Mean` | Gets the mean for each feature computed during fitting. |
| `StdDev` | Gets the standard deviation for each feature computed during fitting. |
| `SupportsInverseTransform` | Gets whether this transformer supports inverse transformation. |

## Methods

| Method | Summary |
|:-----|:--------|
| `FitCore(Matrix<>)` | Computes the mean and standard deviation for each feature from the training data. |
| `GetFeatureNamesOut(String[])` | Gets the output feature names after transformation. |
| `InverseTransformCore(Matrix<>)` | Reverses the global contrast scaling transformation. |
| `TransformCore(Matrix<>)` | Transforms the data by applying global contrast scaling. |

