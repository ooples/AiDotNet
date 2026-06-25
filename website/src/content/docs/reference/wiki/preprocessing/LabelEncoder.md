---
title: "LabelEncoder<T>"
description: "Encodes categorical values as integer labels (0, 1, 2, ...)."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Preprocessing.Encoders`

Encodes categorical values as integer labels (0, 1, 2, ...).

## For Beginners

This encoder converts categories to numbers:

- Each unique value gets a unique number starting from 0
- Values are sorted alphabetically/numerically before encoding

Example:
["cat", "dog", "cat", "bird", "dog"] → [1, 2, 1, 0, 2]
(Mapping: bird=0, cat=1, dog=2)

## How It Works

LabelEncoder transforms categorical values to consecutive integers.
Each unique value is assigned a unique integer starting from 0.
This is useful for encoding target labels or ordinal features.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `LabelEncoder(Int32[])` | Creates a new instance of `LabelEncoder`. |

## Properties

| Property | Summary |
|:-----|:--------|
| `NClasses` | Gets the number of unique classes for each encoded column. |
| `SupportsInverseTransform` | Gets whether this transformer supports inverse transformation. |

## Methods

| Method | Summary |
|:-----|:--------|
| `FitCore(Matrix<>)` | Learns the encoding mapping from the training data. |
| `GetFeatureNamesOut(String[])` | Gets the output feature names after transformation. |
| `InverseTransformCore(Matrix<>)` | Reverses the label encoding to get original values. |
| `TransformCore(Matrix<>)` | Transforms the data by encoding categorical values as integers. |

