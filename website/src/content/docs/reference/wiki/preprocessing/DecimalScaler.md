---
title: "DecimalScaler<T>"
description: "Scales features by dividing by the smallest power of 10 greater than the max absolute value."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Preprocessing.Scalers`

Scales features by dividing by the smallest power of 10 greater than the max absolute value.

## For Beginners

This scaler adjusts numbers to show them in appropriate decimal places:

- If your largest value is 750, it divides everything by 1,000
- So 750 becomes 0.75, 42 becomes 0.042, etc.
- All values end up between -1 and 1

This is useful when you want to keep relative sizes clear and decimal places meaningful.

## How It Works

Decimal scaling transforms values to fall between -1 and 1 by dividing by an appropriate power of 10.
This preserves the relative decimal positions and signs of values.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `DecimalScaler(Int32[])` | Creates a new instance of `DecimalScaler`. |

## Properties

| Property | Summary |
|:-----|:--------|
| `Scale` | Gets the power-of-10 scale factor for each feature computed during fitting. |
| `SupportsInverseTransform` | Gets whether this transformer supports inverse transformation. |

## Methods

| Method | Summary |
|:-----|:--------|
| `FitCore(Matrix<>)` | Computes the power-of-10 scale factor for each feature from the training data. |
| `GetFeatureNamesOut(String[])` | Gets the output feature names after transformation. |
| `InverseTransformCore(Matrix<>)` | Reverses the decimal scaling transformation. |
| `TransformCore(Matrix<>)` | Transforms the data by applying decimal scaling. |

