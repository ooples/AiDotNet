---
title: "ZScoreClipper<T>"
description: "Clips outliers based on Z-score (standard deviation) bounds."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Preprocessing.OutlierHandling`

Clips outliers based on Z-score (standard deviation) bounds.

## For Beginners

Z-score measures how many standard deviations a value is from the mean.

- A Z-score of 0 means the value equals the mean
- A Z-score of 2 means the value is 2 standard deviations above the mean
- A Z-score of -3 means the value is 3 standard deviations below the mean

This clipper replaces extreme values (those with high absolute Z-scores) with
the boundary values, reducing the impact of outliers while preserving data size.

## How It Works

ZScoreClipper identifies outliers as values that deviate from the mean by more than
a specified number of standard deviations. Values beyond this threshold are clipped
to the boundary values.

**How It Works:**
For each feature:

1. Calculate the mean and standard deviation
2. Compute bounds: [mean - threshold*std, mean + threshold*std]
3. Clip values outside these bounds

**Common Thresholds:**

- 2.0: Aggressive (clips ~5% of normally distributed data)
- 3.0: Standard (clips ~0.3% of normally distributed data)
- 3.5: Conservative (clips ~0.05% of normally distributed data)

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `ZScoreClipper(Double,Int32[])` | Creates a new instance of `ZScoreClipper`. |

## Properties

| Property | Summary |
|:-----|:--------|
| `LowerBounds` | Gets the computed lower bounds for each feature. |
| `Means` | Gets the computed means for each feature. |
| `StandardDeviations` | Gets the computed standard deviations for each feature. |
| `SupportsInverseTransform` | Gets whether this transformer supports inverse transformation. |
| `Threshold` | Gets the Z-score threshold for clipping. |
| `UpperBounds` | Gets the computed upper bounds for each feature. |

## Methods

| Method | Summary |
|:-----|:--------|
| `FitCore(Matrix<>)` | Computes the mean and standard deviation for each feature. |
| `GetFeatureNamesOut(String[])` | Gets the output feature names after transformation. |
| `GetOutlierMask(Matrix<>)` | Gets a boolean mask indicating which values are outliers. |
| `GetZScores(Matrix<>)` | Calculates the Z-scores for all values in the data. |
| `InverseTransformCore(Matrix<>)` | Inverse transformation is not supported. |
| `TransformCore(Matrix<>)` | Clips values to the computed Z-score bounds. |

