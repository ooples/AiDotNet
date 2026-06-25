---
title: "Normalizer<T>"
description: "Normalizes samples (rows) individually to unit norm (L1, L2, or Max)."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Preprocessing.Scalers`

Normalizes samples (rows) individually to unit norm (L1, L2, or Max).

## For Beginners

This normalizer scales each row so its "length" equals 1:

- L1 norm: The sum of absolute values equals 1
- L2 norm: The Euclidean length (sqrt of sum of squares) equals 1
- Max norm: The maximum absolute value equals 1

Example with L2 norm: [3, 4] has length 5, so it becomes [0.6, 0.8] (length = 1)

## How It Works

Unlike scalers that operate on columns (features), this normalizer operates on rows (samples).
Each sample is scaled to have a unit norm (length of 1) in the specified norm type.
This is useful when the magnitude of samples varies but their direction matters.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `Normalizer(NormType,Int32[])` | Creates a new instance of `Normalizer`. |

## Properties

| Property | Summary |
|:-----|:--------|
| `NormType` | Gets the norm type used for normalization. |
| `SupportsInverseTransform` | Gets whether this transformer supports inverse transformation. |

## Methods

| Method | Summary |
|:-----|:--------|
| `FitCore(Matrix<>)` | Fits the normalizer to the training data. |
| `GetFeatureNamesOut(String[])` | Gets the output feature names after transformation. |
| `InverseTransformCore(Matrix<>)` | Inverse transformation is not supported for sample normalization. |
| `TransformCore(Matrix<>)` | Transforms the data by normalizing each row to unit norm. |

