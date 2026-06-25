---
title: "Binarizer<T>"
description: "Binarizes features based on a threshold value."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Preprocessing.Discretizers`

Binarizes features based on a threshold value.

## For Beginners

This transformer converts any values to just 0s and 1s:

- If a value is above the threshold → 1
- If a value is at or below the threshold → 0

Example with threshold=5: [3, 6, 2, 8, 5] → [0, 1, 0, 1, 0]

## How It Works

Binarization transforms continuous values to binary (0 or 1) based on a threshold.
Values greater than the threshold become 1, values less than or equal become 0.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `Binarizer(Double,Int32[])` | Creates a new instance of `Binarizer` with a custom threshold. |
| `Binarizer(Int32[])` | Creates a new instance of `Binarizer` with a default threshold of 0. |

## Properties

| Property | Summary |
|:-----|:--------|
| `SupportsInverseTransform` | Gets whether this transformer supports inverse transformation. |
| `Threshold` | Gets the threshold value used for binarization. |

## Methods

| Method | Summary |
|:-----|:--------|
| `FitCore(Matrix<>)` | Fits the binarizer to the training data. |
| `GetFeatureNamesOut(String[])` | Gets the output feature names after transformation. |
| `InverseTransformCore(Matrix<>)` | Inverse transformation is not supported for binarization. |
| `TransformCore(Matrix<>)` | Transforms the data by applying threshold binarization. |

