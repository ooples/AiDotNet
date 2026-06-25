---
title: "OutlierClipper<T>"
description: "Clips outliers to specified percentile bounds."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Preprocessing.OutlierHandling`

Clips outliers to specified percentile bounds.

## For Beginners

Outliers are extreme values that are far from most of your data.
They can distort your model's learning. Clipping replaces extreme values with more
reasonable bounds:

- Values below the 1st percentile → replaced with 1st percentile value
- Values above the 99th percentile → replaced with 99th percentile value

Example: Income data where most people earn $30K-$200K but a few billionaires
would be clipped to prevent the model from being skewed by extreme wealth.

## How It Works

OutlierClipper clips values below a lower percentile and above an upper percentile
to those percentile values. This reduces the impact of extreme outliers while
preserving the overall data distribution.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `OutlierClipper(Double,Double,Int32[])` | Creates a new instance of `OutlierClipper`. |

## Properties

| Property | Summary |
|:-----|:--------|
| `LowerBounds` | Gets the computed lower bounds for each feature. |
| `LowerPercentile` | Gets the lower percentile (values below this are clipped). |
| `SupportsInverseTransform` | Gets whether this transformer supports inverse transformation. |
| `UpperBounds` | Gets the computed upper bounds for each feature. |
| `UpperPercentile` | Gets the upper percentile (values above this are clipped). |

## Methods

| Method | Summary |
|:-----|:--------|
| `FitCore(Matrix<>)` | Computes the percentile bounds for each feature. |
| `GetFeatureNamesOut(String[])` | Gets the output feature names after transformation. |
| `InverseTransformCore(Matrix<>)` | Inverse transformation is not supported. |
| `TransformCore(Matrix<>)` | Clips values to the computed percentile bounds. |

