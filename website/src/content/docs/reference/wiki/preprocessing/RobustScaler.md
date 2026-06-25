---
title: "RobustScaler<T>"
description: "Scales features using statistics that are robust to outliers."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Preprocessing.Scalers`

Scales features using statistics that are robust to outliers.

## For Beginners

This scaler is like StandardScaler but better handles outliers:

- Uses median (middle value) instead of mean (average)
- Uses IQR (spread of middle 50%) instead of standard deviation

Why this matters:

- Mean and std are heavily influenced by extreme values
- Median and IQR ignore extreme values

Example: If most house prices are $100K-$500K but a few are $10M,
RobustScaler won't let those mansions distort the scaling.

## How It Works

Robust scaling removes the median and scales data according to the interquartile range (IQR).
The IQR is the range between the 25th percentile (Q1) and 75th percentile (Q3).
Unlike StandardScaler, RobustScaler uses statistics that are less affected by outliers.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `RobustScaler(Boolean,Boolean,Int32[])` | Creates a new instance of `RobustScaler` with default settings. |
| `RobustScaler(Double,Double,Boolean,Boolean,Int32[])` | Creates a new instance of `RobustScaler` with custom quantile range. |

## Properties

| Property | Summary |
|:-----|:--------|
| `InterquartileRange` | Gets the interquartile range (IQR) of each feature computed during fitting. |
| `Median` | Gets the median of each feature computed during fitting. |
| `SupportsInverseTransform` | Gets whether this transformer supports inverse transformation. |
| `WithCentering` | Gets whether this scaler centers the data (subtracts median). |
| `WithScaling` | Gets whether this scaler scales the data (divides by IQR). |

## Methods

| Method | Summary |
|:-----|:--------|
| `ComputeQuantile([],Double)` | Computes a quantile from sorted data using linear interpolation. |
| `FitCore(Matrix<>)` | Computes the median and IQR of each feature from the training data. |
| `GetFeatureNamesOut(String[])` | Gets the output feature names after transformation. |
| `InverseTransformCore(Matrix<>)` | Reverses the robust scaling transformation. |
| `SortColumn(Vector<>)` | Sorts a column vector for quantile computation. |
| `TransformCore(Matrix<>)` | Transforms the data by applying robust scaling. |

