---
title: "IQRClipper<T>"
description: "Clips outliers using the Interquartile Range (IQR) method."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Preprocessing.OutlierHandling`

Clips outliers using the Interquartile Range (IQR) method.

## For Beginners

The IQR is the range where the middle 50% of data falls.
Values outside 1.5× this range are considered outliers:

- Q1 (25th percentile): 25% of data is below this value
- Q3 (75th percentile): 75% of data is below this value
- IQR = Q3 - Q1: The spread of the middle 50%
- Outliers: Values below Q1 - 1.5×IQR or above Q3 + 1.5×IQR

## How It Works

IQRClipper identifies outliers using the IQR method:

- Lower bound = Q1 - k * IQR
- Upper bound = Q3 + k * IQR

where IQR = Q3 - Q1 and k is the multiplier (default 1.5).

This is the same method used in box plots for identifying outliers.
A multiplier of 1.5 identifies "mild" outliers, while 3.0 identifies "extreme" outliers.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `IQRClipper(Double,Int32[])` | Creates a new instance of `IQRClipper`. |

## Properties

| Property | Summary |
|:-----|:--------|
| `IQRValues` | Gets the IQR values for each feature. |
| `LowerBounds` | Gets the fitted lower bounds for each feature. |
| `Multiplier` | Gets the IQR multiplier for determining outlier boundaries. |
| `Q1Values` | Gets the Q1 (25th percentile) values for each feature. |
| `Q3Values` | Gets the Q3 (75th percentile) values for each feature. |
| `SupportsInverseTransform` | Gets whether this transformer supports inverse transformation. |
| `UpperBounds` | Gets the fitted upper bounds for each feature. |

## Methods

| Method | Summary |
|:-----|:--------|
| `CountOutliersPerFeature(Matrix<>)` | Counts the number of outliers per feature. |
| `FitCore(Matrix<>)` | Fits the clipper by computing IQR bounds for each feature. |
| `GetFeatureNamesOut(String[])` | Gets the output feature names after transformation. |
| `GetOutlierMask(Matrix<>)` | Gets a mask indicating which values are outliers in the input data. |
| `InverseTransformCore(Matrix<>)` | Inverse transformation is not supported. |
| `TransformCore(Matrix<>)` | Transforms the data by clipping values outside IQR bounds. |

