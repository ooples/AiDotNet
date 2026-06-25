---
title: "BackwardDifferenceEncoder<T>"
description: "Encodes categorical features using backward difference coding."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Preprocessing.Encoders`

Encodes categorical features using backward difference coding.

## For Beginners

Backward difference coding is useful for ordinal data:

- Education: High School vs Some College vs Bachelor's vs Master's
- Each coefficient shows the "step up" from the previous level
- Good when you expect gradual progression through levels

## How It Works

BackwardDifferenceEncoder compares each level of a categorical variable to
the previous level. This is useful for ordinal variables where the
difference between adjacent levels is meaningful.

For k categories, creates k-1 columns. Each column represents the difference
between level n and level n-1.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `BackwardDifferenceEncoder(BackwardDifferenceHandleUnknown,Int32[])` | Creates a new instance of `BackwardDifferenceEncoder`. |

## Properties

| Property | Summary |
|:-----|:--------|
| `HandleUnknown` | Gets how unknown categories are handled. |
| `NOutputFeatures` | Gets the number of output features. |
| `SupportsInverseTransform` | Gets whether this transformer supports inverse transformation. |

## Methods

| Method | Summary |
|:-----|:--------|
| `FitCore(Matrix<>)` | Fits the encoder by learning categories and building contrast matrices. |
| `GetFeatureNamesOut(String[])` | Gets the output feature names after transformation. |
| `InverseTransformCore(Matrix<>)` | Inverse transformation is not supported. |
| `TransformCore(Matrix<>)` | Transforms the data using backward difference coding. |

