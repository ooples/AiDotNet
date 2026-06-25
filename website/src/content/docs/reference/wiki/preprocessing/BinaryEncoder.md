---
title: "BinaryEncoder<T>"
description: "Encodes categorical features using binary representation."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Preprocessing.Encoders`

Encodes categorical features using binary representation.

## For Beginners

If you have 8 categories, one-hot creates 8 columns.
Binary encoding uses only 3 columns (since 8 = 2^3):

- Category 0 → [0, 0, 0]
- Category 1 → [0, 0, 1]
- Category 2 → [0, 1, 0]
- Category 7 → [1, 1, 1]

This is useful for high-cardinality categorical features where one-hot
would create too many columns.

## How It Works

BinaryEncoder first ordinal encodes the categories, then converts those integers
to their binary representation. This creates log2(n) columns instead of n columns
that one-hot encoding would create.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `BinaryEncoder(BinaryEncoderHandleUnknown,Int32[])` | Creates a new instance of `BinaryEncoder`. |

## Properties

| Property | Summary |
|:-----|:--------|
| `HandleUnknown` | Gets how unknown categories are handled. |
| `NOutputFeatures` | Gets the number of output features after transformation. |
| `SupportsInverseTransform` | Gets whether this transformer supports inverse transformation. |

## Methods

| Method | Summary |
|:-----|:--------|
| `FitCore(Matrix<>)` | Learns the categories and binary encoding from the training data. |
| `GetFeatureNamesOut(String[])` | Gets the output feature names after transformation. |
| `InverseTransformCore(Matrix<>)` | Reverses the binary encoding to get original category values. |
| `TransformCore(Matrix<>)` | Transforms the data by converting categories to binary representation. |

