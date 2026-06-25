---
title: "OneHotEncoder<T>"
description: "Encodes categorical values as one-hot (binary) vectors."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Preprocessing.Encoders`

Encodes categorical values as one-hot (binary) vectors.

## For Beginners

This encoder converts categories into binary columns:

- Each unique value gets its own column
- A 1 indicates the category is present, 0 means it's not

Example for colors [red, green, blue, red]:
Becomes:
[1, 0, 0] (red)
[0, 1, 0] (green)
[0, 0, 1] (blue)
[1, 0, 0] (red)

## How It Works

OneHotEncoder transforms categorical values into binary indicator columns.
Each unique category value becomes a separate column with 1s and 0s indicating presence.
This encoding is required for many machine learning algorithms that cannot work directly with categories.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `OneHotEncoder(Boolean,OneHotUnknownHandling,Int32[])` | Creates a new instance of `OneHotEncoder`. |

## Properties

| Property | Summary |
|:-----|:--------|
| `Categories` | Gets the categories for each encoded column. |
| `DropFirst` | Gets whether the first category is dropped (to avoid multicollinearity). |
| `HandleUnknown` | Gets how unknown categories are handled. |
| `NOutputFeatures` | Gets the number of output features after transformation. |
| `SupportsInverseTransform` | Gets whether this transformer supports inverse transformation. |

## Methods

| Method | Summary |
|:-----|:--------|
| `FitCore(Matrix<>)` | Learns the categories from the training data. |
| `GetFeatureNamesOut(String[])` | Gets the output feature names after transformation. |
| `InverseTransformCore(Matrix<>)` | Reverses the one-hot encoding to get original category values. |
| `TransformCore(Matrix<>)` | Transforms the data by applying one-hot encoding. |

