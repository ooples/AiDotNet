---
title: "CatBoostEncoder<T>"
description: "Encodes categorical features using ordered (CatBoost-style) target encoding."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Preprocessing.Encoders`

Encodes categorical features using ordered (CatBoost-style) target encoding.

## For Beginners

Regular target encoding can "cheat" by using future
information. CatBoost encoding prevents this:

- When encoding row 10, it only uses data from rows 1-9
- Row 1 always gets the prior (global mean) since there's nothing before it
- This prevents overfitting and works better with gradient boosting

## How It Works

CatBoostEncoder applies an ordered approach to target encoding that prevents
target leakage by only using target values from previous samples when encoding.
This is the same technique used in the CatBoost gradient boosting library.

For each sample, the encoding is computed as:
(sum of targets for previous samples with same category + prior) / (count + 1)

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `CatBoostEncoder(Double,CatBoostHandleUnknown,Int32,Int32[])` | Creates a new instance of `CatBoostEncoder`. |

## Properties

| Property | Summary |
|:-----|:--------|
| `GlobalMean` | Gets the global target mean. |
| `HandleUnknown` | Gets how unknown categories are handled. |
| `Prior` | Gets the prior value (regularization). |
| `SupportsInverseTransform` | Gets whether this transformer supports inverse transformation. |

## Methods

| Method | Summary |
|:-----|:--------|
| `Fit(Matrix<>,Vector<>)` | Fits the encoder by computing category statistics. |
| `FitCore(Matrix<>)` | Fits the encoder (requires target via specialized Fit method). |
| `FitTransform(Matrix<>,Vector<>)` | Fits and transforms using ordered target encoding (CatBoost style). |
| `GetFeatureNamesOut(String[])` | Gets the output feature names after transformation. |
| `InverseTransformCore(Matrix<>)` | Inverse transformation is not supported. |
| `TransformCore(Matrix<>)` | Transforms test data using full category statistics (for inference). |
| `TransformWithTarget(Matrix<>,Vector<>)` | Transforms training data using ordered encoding (only uses previous samples). |

