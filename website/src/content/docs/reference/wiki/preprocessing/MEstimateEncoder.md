---
title: "MEstimateEncoder<T>"
description: "Encodes categorical features using M-estimate regularization."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Preprocessing.Encoders`

Encodes categorical features using M-estimate regularization.

## For Beginners

M-estimate is like adding 'm' fake samples:

- Each fake sample has the global mean as its target
- Categories with few samples get pulled toward the global mean
- Categories with many samples stay close to their actual mean
- Higher m = more smoothing toward global mean

## How It Works

MEstimateEncoder applies M-estimate smoothing to target encoding, which adds
a regularization parameter 'm' that controls shrinkage toward the global mean.

The formula: encoded = (n * category_mean + m * global_mean) / (n + m)
where n is the count of samples in the category and m is the smoothing parameter.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `MEstimateEncoder(Double,MEstimateHandleUnknown,Int32[])` | Creates a new instance of `MEstimateEncoder`. |

## Properties

| Property | Summary |
|:-----|:--------|
| `GlobalMean` | Gets the global target mean. |
| `HandleUnknown` | Gets how unknown categories are handled. |
| `M` | Gets the smoothing parameter m. |
| `SupportsInverseTransform` | Gets whether this transformer supports inverse transformation. |

## Methods

| Method | Summary |
|:-----|:--------|
| `Fit(Matrix<>,Vector<>)` | Fits the encoder by computing M-estimate encodings. |
| `FitCore(Matrix<>)` | Fits the encoder (requires target via specialized Fit method). |
| `FitTransform(Matrix<>,Vector<>)` | Fits and transforms the data. |
| `GetFeatureNamesOut(String[])` | Gets the output feature names after transformation. |
| `InverseTransformCore(Matrix<>)` | Inverse transformation is not supported. |
| `TransformCore(Matrix<>)` | Transforms the data using fitted encodings. |

