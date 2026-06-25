---
title: "TargetEncoder<T>"
description: "Encodes categorical features using target mean encoding."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Preprocessing.Encoders`

Encodes categorical features using target mean encoding.

## For Beginners

Instead of one-hot encoding (many columns), target encoding
creates a single column per feature containing the average target value for each category:

- Category "A" with average target 0.8 becomes 0.8
- Category "B" with average target 0.3 becomes 0.3

This is especially useful for high-cardinality features where one-hot would create
too many columns.

## How It Works

TargetEncoder replaces each category with the mean of the target variable for that category.
This creates a continuous feature that captures the relationship between the category and target.

To prevent overfitting, especially with rare categories, smoothing is applied:
encoding = (count * category_mean + smoothing * global_mean) / (count + smoothing)

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `TargetEncoder(Double,Double,TargetEncoderHandleUnknown,Int32[])` | Creates a new instance of `TargetEncoder`. |

## Properties

| Property | Summary |
|:-----|:--------|
| `EncodingMaps` | Gets the encoding maps for each column. |
| `HandleUnknown` | Gets how unknown categories are handled during transform. |
| `Smoothing` | Gets the smoothing parameter used during encoding. |
| `SupportsInverseTransform` | Gets whether this transformer supports inverse transformation. |

## Methods

| Method | Summary |
|:-----|:--------|
| `Fit(Matrix<>,Vector<>)` | Fits the encoder by learning the target means for each category. |
| `FitCore(Matrix<>)` | Fits the encoder using the base Fit method (requires target via FitWithTarget). |
| `FitTransform(Matrix<>,Vector<>)` | Fits the encoder and transforms the data in one step. |
| `GetFeatureNamesOut(String[])` | Gets the output feature names after transformation. |
| `InverseTransformCore(Matrix<>)` | Inverse transformation is not supported for target encoding. |
| `TransformCore(Matrix<>)` | Transforms the data by replacing categories with their target means. |

