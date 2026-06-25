---
title: "LogScaler<T>"
description: "Applies logarithmic transformation to features, useful for data spanning multiple orders of magnitude."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Preprocessing.Scalers`

Applies logarithmic transformation to features, useful for data spanning multiple orders of magnitude.

## For Beginners

Log normalization measures percentages rather than absolute amounts:

- With regular measurement, going from 1 to 10 and from 10 to 100 look very different
- With logarithmic measurement, both represent a "10× increase" and appear as equal steps

Example: [1,000, 10,000, 100,000, 1,000,000] becomes evenly spaced values.

## How It Works

Log scaling transforms data using natural logarithm, which compresses the range of values.
It shifts negative values to ensure all inputs are positive, then applies log scaling.
This is particularly useful for exponentially distributed data.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `LogScaler(Int32[])` | Creates a new instance of `LogScaler`. |

## Properties

| Property | Summary |
|:-----|:--------|
| `Shift` | Gets the shift applied to each feature to ensure positive values. |
| `SupportsInverseTransform` | Gets whether this transformer supports inverse transformation. |

## Methods

| Method | Summary |
|:-----|:--------|
| `FitCore(Matrix<>)` | Computes the shift and log range for each feature from the training data. |
| `GetFeatureNamesOut(String[])` | Gets the output feature names after transformation. |
| `InverseTransformCore(Matrix<>)` | Reverses the log scaling transformation. |
| `TransformCore(Matrix<>)` | Transforms the data by applying log scaling. |

