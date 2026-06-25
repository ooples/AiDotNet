---
title: "LeaveOneOutEncoder<T>"
description: "Encodes categorical features using leave-one-out target encoding."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Preprocessing.Encoders`

Encodes categorical features using leave-one-out target encoding.

## For Beginners

Regular target encoding can overfit because it uses
the same data to encode and train. Leave-one-out encoding prevents this:

- When encoding row 1, it uses the average of all OTHER rows with the same category
- This prevents the model from "cheating" by memorizing individual samples

Example: If "Category A" has 3 samples with targets [1, 0, 1]:

- Row 1 gets encoded as average of [0, 1] = 0.5
- Row 2 gets encoded as average of [1, 1] = 1.0
- Row 3 gets encoded as average of [1, 0] = 0.5

## How It Works

LeaveOneOutEncoder is similar to TargetEncoder but uses leave-one-out statistics
to prevent overfitting. For each sample, the encoding is computed using all other
samples in the same category, excluding the current sample.

This reduces the risk of target leakage during training while still capturing
the relationship between categories and the target variable.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `LeaveOneOutEncoder(Double,LeaveOneOutHandleUnknown,Int32[])` | Creates a new instance of `LeaveOneOutEncoder`. |

## Properties

| Property | Summary |
|:-----|:--------|
| `GlobalMean` | Gets the global target mean. |
| `HandleUnknown` | Gets how unknown categories are handled. |
| `Smoothing` | Gets the smoothing parameter. |
| `SupportsInverseTransform` | Gets whether this transformer supports inverse transformation. |

## Methods

| Method | Summary |
|:-----|:--------|
| `Fit(Matrix<>,Vector<>)` | Fits the encoder by computing category statistics. |
| `FitCore(Matrix<>)` | Fits the encoder (requires target via specialized Fit method). |
| `FitTransform(Matrix<>,Vector<>)` | Fits and transforms using leave-one-out encoding. |
| `GetFeatureNamesOut(String[])` | Gets the output feature names after transformation. |
| `InverseTransformCore(Matrix<>)` | Inverse transformation is not supported. |
| `TransformCore(Matrix<>)` | Transforms the data using standard target encoding (for test/inference data). |
| `TransformWithTarget(Matrix<>,Vector<>)` | Transforms the data using leave-one-out encoding (requires target for training data). |

