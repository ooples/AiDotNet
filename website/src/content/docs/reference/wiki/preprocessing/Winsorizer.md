---
title: "Winsorizer<T>"
description: "Winsorizes data by replacing extreme values with percentile bounds."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Preprocessing.OutlierHandling`

Winsorizes data by replacing extreme values with percentile bounds.

## For Beginners

Winsorization is named after biostatistician Charles Winsor.
Instead of removing outliers, it replaces them with the nearest "normal" values:

- If you Winsorize at 5%, the bottom 5% of values become equal to the 5th percentile
- The top 5% of values become equal to the 95th percentile

This preserves sample size while reducing outlier impact.

## How It Works

Winsorizer is a statistical technique that limits extreme values in the data
to reduce the effect of outliers. Unlike trimming (which removes outliers),
Winsorization replaces them with less extreme values.

This is equivalent to OutlierClipper but follows the traditional Winsorization
terminology where you specify the percentage of data to Winsorize at each tail.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `Winsorizer(Double,Double,WinsorizerLimitType,Int32[])` | Creates a new instance of `Winsorizer`. |

## Properties

| Property | Summary |
|:-----|:--------|
| `LimitType` | Gets the type of limit (percentile or IQR). |
| `LowerBounds` | Gets the computed lower bounds for each feature. |
| `LowerLimit` | Gets the lower limit value. |
| `SupportsInverseTransform` | Gets whether this transformer supports inverse transformation. |
| `UpperBounds` | Gets the computed upper bounds for each feature. |
| `UpperLimit` | Gets the upper limit value. |

## Methods

| Method | Summary |
|:-----|:--------|
| `FitCore(Matrix<>)` | Computes the Winsorization bounds for each feature. |
| `GetFeatureNamesOut(String[])` | Gets the output feature names after transformation. |
| `InverseTransformCore(Matrix<>)` | Inverse transformation is not supported. |
| `TransformCore(Matrix<>)` | Winsorizes the data by replacing extreme values with bounds. |

