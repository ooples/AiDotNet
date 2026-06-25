---
title: "StandardScaler<T>"
description: "Standardizes features by removing the mean and scaling to unit variance."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Preprocessing.Scalers`

Standardizes features by removing the mean and scaling to unit variance.

## For Beginners

This scaler converts your data to a standard scale:

- The center of your data (mean) becomes 0
- The spread of your data (standard deviation) becomes 1

This is like converting different currencies to a common one - it makes
different features comparable and helps many ML algorithms work better.

## How It Works

Standard scaling (Z-score normalization) transforms data to have a mean of 0 and a
standard deviation of 1. This is important for many machine learning algorithms as
it puts different features on comparable scales.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `StandardScaler(Boolean,Boolean,Int32[])` | Creates a new instance of `StandardScaler`. |

## Properties

| Property | Summary |
|:-----|:--------|
| `Mean` | Gets the mean of each feature computed during fitting. |
| `StandardDeviation` | Gets the standard deviation of each feature computed during fitting. |
| `SupportsInverseTransform` | Gets whether this transformer supports inverse transformation. |
| `WithMean` | Gets whether this scaler centers the data (subtracts mean). |
| `WithStd` | Gets whether this scaler scales the data (divides by std). |

## Methods

| Method | Summary |
|:-----|:--------|
| `FitCore(Matrix<>)` | Computes the mean and standard deviation of each feature from the training data. |
| `GetFeatureNamesOut(String[])` | Gets the output feature names after transformation. |
| `InverseTransformCore(Matrix<>)` | Reverses the standardization transformation. |
| `TransformCore(Matrix<>)` | Transforms the data by applying the computed mean and standard deviation. |

