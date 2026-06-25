---
title: "ThresholdClipper<T>"
description: "Clips values based on explicit threshold bounds."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Preprocessing.OutlierHandling`

Clips values based on explicit threshold bounds.

## For Beginners

This is the simplest outlier clipper - you tell it exactly what
the minimum and maximum allowed values are, and it clips anything outside those bounds.
This is useful when you know from domain knowledge what the valid range should be.

## How It Works

ThresholdClipper clips values to user-specified lower and upper bounds. Unlike other
clippers that compute bounds from data statistics, this clipper uses explicit thresholds
provided by the user.

**Use Cases:**

- Domain-specific constraints (e.g., percentages must be 0-100)
- Physical limits (e.g., temperatures can't be below absolute zero)
- Business rules (e.g., prices must be positive)
- Known valid ranges from domain expertise

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `ThresholdClipper(Double,Double,Int32[])` | Creates a new instance of `ThresholdClipper` with explicit bounds. |
| `ThresholdClipper(Double,Int32[])` | Creates a new instance of `ThresholdClipper` with symmetric bounds. |

## Properties

| Property | Summary |
|:-----|:--------|
| `LowerThreshold` | Gets the lower threshold bound. |
| `SupportsInverseTransform` | Gets whether this transformer supports inverse transformation. |
| `UpperThreshold` | Gets the upper threshold bound. |

## Methods

| Method | Summary |
|:-----|:--------|
| `CountOutliers(Matrix<>)` | Counts how many values would be clipped at each bound. |
| `FitCore(Matrix<>)` | Records the number of input features. |
| `GetFeatureNamesOut(String[])` | Gets the output feature names after transformation. |
| `GetOutlierMask(Matrix<>)` | Gets a boolean mask indicating which values are outside the threshold bounds. |
| `InverseTransformCore(Matrix<>)` | Inverse transformation is not supported. |
| `TransformCore(Matrix<>)` | Clips values to the explicit threshold bounds. |

