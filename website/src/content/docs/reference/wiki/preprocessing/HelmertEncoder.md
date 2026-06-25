---
title: "HelmertEncoder<T>"
description: "Encodes categorical features using Helmert coding."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Preprocessing.Encoders`

Encodes categorical features using Helmert coding.

## For Beginners

Helmert coding is useful when order matters:

- Compares each level to the "future average"
- First level vs. average of all others
- Second level vs. average of third, fourth, etc.
- Good for detecting trends or cumulative effects

## How It Works

HelmertEncoder compares each level of a categorical variable to the mean of
all subsequent levels. This is useful when you want to understand how each
level differs from the average of all levels that come after it.

For k categories, creates k-1 columns. Column i compares level i to the
mean of levels i+1 through k.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `HelmertEncoder(HelmertHandleUnknown,Boolean,Int32[])` | Creates a new instance of `HelmertEncoder`. |

## Properties

| Property | Summary |
|:-----|:--------|
| `HandleUnknown` | Gets how unknown categories are handled. |
| `NOutputFeatures` | Gets the number of output features. |
| `Reversed` | Gets whether reversed Helmert coding is used. |
| `SupportsInverseTransform` | Gets whether this transformer supports inverse transformation. |

## Methods

| Method | Summary |
|:-----|:--------|
| `FitCore(Matrix<>)` | Fits the encoder by learning categories and building contrast matrices. |
| `GetFeatureNamesOut(String[])` | Gets the output feature names after transformation. |
| `InverseTransformCore(Matrix<>)` | Inverse transformation is not supported. |
| `TransformCore(Matrix<>)` | Transforms the data using Helmert coding. |

