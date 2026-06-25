---
title: "QuantileTransformer<T>"
description: "Transforms features to follow a uniform or normal distribution using quantile information."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Preprocessing.PowerTransforms`

Transforms features to follow a uniform or normal distribution using quantile information.

## For Beginners

This transformer redistributes your data to match a desired pattern:

- Uniform: Spreads values evenly across [0, 1]
- Normal: Creates a bell curve distribution

Example with uniform output:
[1, 1, 2, 3, 5, 8, 13, 100, 1000] → [0.0, 0.125, 0.25, 0.375, 0.5, 0.625, 0.75, 0.875, 1.0]
Notice how extreme values (100, 1000) no longer dominate.

## How It Works

QuantileTransformer applies a non-linear transformation that maps the input distribution to
either a uniform or normal (Gaussian) distribution. This is effective at reducing the impact
of outliers and normalizing non-Gaussian distributions.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `QuantileTransformer(OutputDistributionType,Int32,Int32[])` | Creates a new instance of `QuantileTransformer`. |

## Properties

| Property | Summary |
|:-----|:--------|
| `NQuantiles` | Gets the number of quantiles used. |
| `OutputDistribution` | Gets the target output distribution. |
| `Quantiles` | Gets the quantiles for each feature computed during fitting. |
| `SupportsInverseTransform` | Gets whether this transformer supports inverse transformation. |

## Methods

| Method | Summary |
|:-----|:--------|
| `FitCore(Matrix<>)` | Computes the quantiles for each feature from the training data. |
| `GetFeatureNamesOut(String[])` | Gets the output feature names after transformation. |
| `InverseTransformCore(Matrix<>)` | Reverses the quantile transformation. |
| `TransformCore(Matrix<>)` | Transforms the data by mapping to the target distribution. |

