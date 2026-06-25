---
title: "MADClipper<T>"
description: "Clips outliers based on Median Absolute Deviation (MAD) bounds."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Preprocessing.OutlierHandling`

Clips outliers based on Median Absolute Deviation (MAD) bounds.

## For Beginners

Think of the median as the "middle value" when you sort your data.
MAD measures how spread out your data is from this middle value. Unlike the average,
the median isn't pulled toward extreme values, so MAD gives a more reliable measure
of spread when outliers are present.

## How It Works

MADClipper identifies outliers using the Median Absolute Deviation, which is more robust
to outliers than standard deviation-based methods. Values with modified Z-scores exceeding
the threshold are clipped to the boundary values.

**How It Works:**
For each feature:

1. Calculate the median
2. Calculate MAD = median(|x - median|)
3. Calculate modified Z-score = 0.6745 * (x - median) / MAD
4. Clip values where |modified Z-score| > threshold

**Why MAD is Better Than Z-Score for Outliers:**
The traditional Z-score uses mean and standard deviation, which are themselves
affected by outliers. MAD uses medians, which are resistant to extreme values.
This makes MAD better at detecting outliers when your data already contains
significant outliers.

**Common Thresholds:**

- 2.5: Aggressive (more values treated as outliers)
- 3.5: Standard (recommended default)
- 5.0: Conservative (only extreme outliers)

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `MADClipper(Double,Int32[])` | Creates a new instance of `MADClipper`. |

## Properties

| Property | Summary |
|:-----|:--------|
| `LowerBounds` | Gets the computed lower bounds for each feature. |
| `MADs` | Gets the computed MAD values for each feature. |
| `Medians` | Gets the computed medians for each feature. |
| `SupportsInverseTransform` | Gets whether this transformer supports inverse transformation. |
| `Threshold` | Gets the modified Z-score threshold for clipping. |
| `UpperBounds` | Gets the computed upper bounds for each feature. |

## Methods

| Method | Summary |
|:-----|:--------|
| `FitCore(Matrix<>)` | Computes the median and MAD for each feature. |
| `GetFeatureNamesOut(String[])` | Gets the output feature names after transformation. |
| `GetModifiedZScores(Matrix<>)` | Calculates the modified Z-scores for all values in the data. |
| `GetOutlierMask(Matrix<>)` | Gets a boolean mask indicating which values are outliers. |
| `InverseTransformCore(Matrix<>)` | Inverse transformation is not supported. |
| `TransformCore(Matrix<>)` | Clips values to the computed MAD bounds. |

## Fields

| Field | Summary |
|:-----|:--------|
| `MADScaleFactor` | Constant for converting MAD to standard deviation equivalent (1 / 0.6745). |

