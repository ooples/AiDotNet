---
title: "OrdinalEncoder<T>"
description: "Encodes categorical values as ordinal integers with optional custom ordering."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Preprocessing.Encoders`

Encodes categorical values as ordinal integers with optional custom ordering.

## For Beginners

This encoder converts categories to ordered numbers:

- You can specify the order of categories
- Useful when categories have a natural ordering (e.g., low, medium, high)

Example with custom order ["small", "medium", "large"]:
["large", "small", "medium", "large"] → [2, 0, 1, 2]

## How It Works

OrdinalEncoder transforms categorical values to consecutive integers based on order.
Unlike LabelEncoder, it can accept custom category orderings and handle unknown values.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `OrdinalEncoder(List<Double[]>,UnknownValueHandling,Double,Int32[])` | Creates a new instance of `OrdinalEncoder`. |

## Properties

| Property | Summary |
|:-----|:--------|
| `HandleUnknown` | Gets how unknown categories are handled. |
| `SupportsInverseTransform` | Gets whether this transformer supports inverse transformation. |
| `UnknownValue` | Gets the value used for unknown categories. |

## Methods

| Method | Summary |
|:-----|:--------|
| `FitCore(Matrix<>)` | Learns the encoding mapping from the training data or uses provided categories. |
| `GetFeatureNamesOut(String[])` | Gets the output feature names after transformation. |
| `InverseTransformCore(Matrix<>)` | Reverses the ordinal encoding to get original values. |
| `TransformCore(Matrix<>)` | Transforms the data by encoding categorical values as ordinal integers. |

