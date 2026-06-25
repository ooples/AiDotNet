---
title: "PowerTransformer<T>"
description: "Applies power transformations (Box-Cox or Yeo-Johnson) to make data more Gaussian-like."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Preprocessing.PowerTransforms`

Applies power transformations (Box-Cox or Yeo-Johnson) to make data more Gaussian-like.

## For Beginners

This transformer makes your data more "bell-curve shaped":

- Box-Cox: For strictly positive data (prices, counts)
- Yeo-Johnson: Works with any data including negatives and zeros

After transformation, features will have more normal (Gaussian) distributions,
which helps many machine learning algorithms perform better.

## How It Works

PowerTransformer applies a power transformation to each feature to make it more Gaussian-like.
Box-Cox requires strictly positive data, while Yeo-Johnson works with any values.
The transformation can help stabilize variance and improve the fit of models that assume normality.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `PowerTransformer(PowerTransformMethod,Boolean,Int32[])` | Creates a new instance of `PowerTransformer`. |

## Properties

| Property | Summary |
|:-----|:--------|
| `Lambdas` | Gets the optimal lambda parameters for each feature. |
| `Method` | Gets the power transformation method used. |
| `Standardize` | Gets whether standardization is applied after transformation. |
| `SupportsInverseTransform` | Gets whether this transformer supports inverse transformation. |

## Methods

| Method | Summary |
|:-----|:--------|
| `FitCore(Matrix<>)` | Learns the optimal lambda parameters for each feature. |
| `GetFeatureNamesOut(String[])` | Gets the output feature names after transformation. |
| `InverseTransformCore(Matrix<>)` | Reverses the power transformation. |
| `TransformCore(Matrix<>)` | Transforms the data by applying the power transformation. |

