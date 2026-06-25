---
title: "SumEncoder<T>"
description: "Encodes categorical features using sum (deviation) coding."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Preprocessing.Encoders`

Encodes categorical features using sum (deviation) coding.

## For Beginners

Sum coding is useful in ANOVA-style analysis:

- Shows how each category differs from the overall average
- The reference category's effect is the negative sum of others
- Coefficients are easier to interpret as deviations from mean

## How It Works

SumEncoder (also known as deviation coding or effect coding) compares each level
to the grand mean of the dependent variable. The last category serves as the
reference and is encoded as -1 in all columns.

For k categories, creates k-1 columns. Unlike one-hot encoding:

- Each category (except reference) gets 1 in its column, 0 elsewhere
- The reference category gets -1 in ALL columns
- The sum of coefficients equals 0

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `SumEncoder(SumEncoderHandleUnknown,Int32[])` | Creates a new instance of `SumEncoder`. |

## Properties

| Property | Summary |
|:-----|:--------|
| `HandleUnknown` | Gets how unknown categories are handled. |
| `NOutputFeatures` | Gets the number of output features. |
| `SupportsInverseTransform` | Gets whether this transformer supports inverse transformation. |

## Methods

| Method | Summary |
|:-----|:--------|
| `FitCore(Matrix<>)` | Fits the encoder by learning categories. |
| `GetFeatureNamesOut(String[])` | Gets the output feature names after transformation. |
| `InverseTransformCore(Matrix<>)` | Inverse transformation is not supported. |
| `TransformCore(Matrix<>)` | Transforms the data using sum coding. |

