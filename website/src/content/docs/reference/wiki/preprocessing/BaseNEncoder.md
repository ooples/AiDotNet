---
title: "BaseNEncoder<T>"
description: "Encodes categorical features using base-N representation."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Preprocessing.Encoders`

Encodes categorical features using base-N representation.

## For Beginners

BaseNEncoder is like counting in different number systems:

- Base 2 (binary): Uses 0 and 1 → most compact
- Base 3 (ternary): Uses 0, 1, 2 → slightly more columns
- Higher bases = fewer columns but more possible values per column

## How It Works

BaseNEncoder converts category indices to base-N representation, creating
multiple columns with digits in the specified base. This is a generalization
of binary encoding (base 2).

For example, with base=3 and 9 categories (0-8):

- Category 0 → [0, 0]
- Category 4 → [1, 1] (4 = 1*3 + 1)
- Category 8 → [2, 2] (8 = 2*3 + 2)

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `BaseNEncoder(Int32,BaseNHandleUnknown,Int32[])` | Creates a new instance of `BaseNEncoder`. |

## Properties

| Property | Summary |
|:-----|:--------|
| `Base` | Gets the base used for encoding. |
| `HandleUnknown` | Gets how unknown categories are handled. |
| `NOutputFeatures` | Gets the number of output features. |
| `SupportsInverseTransform` | Gets whether this transformer supports inverse transformation. |

## Methods

| Method | Summary |
|:-----|:--------|
| `FitCore(Matrix<>)` | Fits the encoder by learning categories. |
| `GetFeatureNamesOut(String[])` | Gets the output feature names after transformation. |
| `InverseTransformCore(Matrix<>)` | Inverse transformation is not supported. |
| `TransformCore(Matrix<>)` | Transforms the data using base-N encoding. |

