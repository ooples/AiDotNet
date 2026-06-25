---
title: "LpNormScaler<T>"
description: "Scales features (columns) by dividing each element by the column's Lp-norm."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Preprocessing.Scalers`

Scales features (columns) by dividing each element by the column's Lp-norm.

## For Beginners

This scaler standardizes each feature column to have a consistent "length".

Think of each column as an arrow (vector) in space:

- The Lp-norm measures the "length" of this arrow
- This scaler divides each element by the length
- The result is an arrow pointing in the same direction but with length = 1

Different p values provide different ways to measure length:

- p = 1: Like measuring distance by walking along city blocks
- p = 2: Like measuring distance "as the crow flies" (straight line)
- Higher p values: Increasingly emphasize the largest component

For example, normalizing the column [3, 4] with p = 2 (Euclidean norm):

- The norm is sqrt(3^2 + 4^2) = sqrt(25) = 5
- The normalized column is [3/5, 4/5] = [0.6, 0.8]
- This new column has length 1 using the L2 norm

This is useful for:

- Feature normalization in machine learning models
- Ensuring consistent scaling across feature vectors
- Applications where feature direction matters more than magnitude

## How It Works

The LpNormScaler normalizes each feature (column) by dividing every element by the
column's Lp-norm. This results in each column having a unit Lp-norm. The Lp-norm is a
generalization of different vector norms based on the parameter p:

- p = 1: Manhattan (L1) norm (sum of absolute values)
- p = 2: Euclidean (L2) norm (square root of sum of squares)
- p = infinity: Maximum (L-infinity) norm (maximum absolute value)

Note: This scaler operates on columns (features), which is different from the Normalizer class
that operates on rows (samples). Use this when you want to normalize feature vectors;
use Normalizer when you want to normalize sample vectors.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `LpNormScaler(Double,Int32[])` | Creates a new instance of `LpNormScaler`. |

## Properties

| Property | Summary |
|:-----|:--------|
| `ColumnNorms` | Gets the Lp-norm of each column computed during fitting. |
| `P` | Gets the p parameter that defines which Lp-norm to use. |
| `SupportsInverseTransform` | Gets whether this transformer supports inverse transformation. |

## Methods

| Method | Summary |
|:-----|:--------|
| `FitCore(Matrix<>)` | Computes the Lp-norm of each column. |
| `GetFeatureNamesOut(String[])` | Gets the output feature names after transformation. |
| `InverseTransformCore(Matrix<>)` | Reverses the Lp-norm scaling by multiplying each column by its original norm. |
| `TransformCore(Matrix<>)` | Transforms the data by dividing each column by its Lp-norm. |

