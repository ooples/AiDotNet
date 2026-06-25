---
title: "LogMeanVarianceScaler<T>"
description: "Applies logarithmic transformation followed by mean-variance standardization."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Preprocessing.Scalers`

Applies logarithmic transformation followed by mean-variance standardization.

## For Beginners

This scaler is perfect for highly skewed or exponentially distributed data:

- First, it takes the log of values (compressing large differences)
- Then, it standardizes to zero mean and unit variance

Example: [1000, 10000, 100000, 1000000] → after log: [6.9, 9.2, 11.5, 13.8] → standardized
This makes patterns in exponential data much easier to detect.

## How It Works

Log-mean-variance scaling combines logarithmic transformation with z-score standardization.
It first applies log transformation (with shift for negative values), then standardizes
using mean and standard deviation. This is ideal for data spanning multiple orders of magnitude.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `LogMeanVarianceScaler(Int32[])` | Creates a new instance of `LogMeanVarianceScaler`. |

## Properties

| Property | Summary |
|:-----|:--------|
| `LogMean` | Gets the mean of log-transformed values for each feature. |
| `LogStdDev` | Gets the standard deviation of log-transformed values for each feature. |
| `Shift` | Gets the shift applied to each feature to ensure positive values. |
| `SupportsInverseTransform` | Gets whether this transformer supports inverse transformation. |

## Methods

| Method | Summary |
|:-----|:--------|
| `FitCore(Matrix<>)` | Computes the shift, log mean, and log standard deviation for each feature. |
| `GetFeatureNamesOut(String[])` | Gets the output feature names after transformation. |
| `InverseTransformCore(Matrix<>)` | Reverses the log-mean-variance scaling transformation. |
| `TransformCore(Matrix<>)` | Transforms the data by applying log transformation and standardization. |

