---
title: "JamesSteinEncoder<T>"
description: "Encodes categorical features using James-Stein shrinkage estimation."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Preprocessing.Encoders`

Encodes categorical features using James-Stein shrinkage estimation.

## For Beginners

This encoder balances between:

- Trusting category-specific averages (when we have lots of data)
- Falling back to the overall average (when category data is sparse)
- The balance is determined automatically using statistical theory

## How It Works

JamesSteinEncoder uses Bayesian shrinkage to blend category-specific target means
with the global mean. Categories with more samples get weights closer to their
own mean, while rare categories shrink toward the global mean.

The shrinkage formula: encoded = (1 - B) * category_mean + B * global_mean
where B is the shrinkage factor based on sample size and variance.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `JamesSteinEncoder(JamesSteinHandleUnknown,Int32[])` | Creates a new instance of `JamesSteinEncoder`. |

## Properties

| Property | Summary |
|:-----|:--------|
| `GlobalMean` | Gets the global target mean. |
| `HandleUnknown` | Gets how unknown categories are handled. |
| `SupportsInverseTransform` | Gets whether this transformer supports inverse transformation. |

## Methods

| Method | Summary |
|:-----|:--------|
| `Fit(Matrix<>,Vector<>)` | Fits the encoder by computing James-Stein shrinkage estimates. |
| `FitCore(Matrix<>)` | Fits the encoder (requires target via specialized Fit method). |
| `FitTransform(Matrix<>,Vector<>)` | Fits and transforms the data. |
| `GetFeatureNamesOut(String[])` | Gets the output feature names after transformation. |
| `InverseTransformCore(Matrix<>)` | Inverse transformation is not supported. |
| `TransformCore(Matrix<>)` | Transforms the data using fitted encodings. |

