---
title: "NormalizationMethod"
description: "Defines different methods for normalizing data before processing in machine learning algorithms."
section: "API Reference"
---

`Enums` · `AiDotNet.Enums`

Defines different methods for normalizing data before processing in machine learning algorithms.

## For Beginners

Normalization is like converting different measurements to a common scale. 
Imagine you have data about people's heights (in feet) and weights (in pounds) - these numbers 
are on very different scales. Normalization transforms all your data to similar ranges (like 0-1) 
so that one feature doesn't overwhelm others just because it uses bigger numbers. This helps 
machine learning algorithms work better and faster.

## Fields

| Field | Summary |
|:-----|:--------|
| `Binning` | Groups continuous data into discrete bins or categories. |
| `Decimal` | Scales values by dividing by powers of 10 to bring all values between 0 and 1. |
| `GlobalContrast` | Normalizes data by adjusting contrast across the entire dataset. |
| `Log` | Applies a logarithmic transformation to compress the range of values. |
| `LogMeanVariance` | Applies logarithmic transformation followed by mean-variance normalization. |
| `LpNorm` | Normalizes data using the Lp norm (typically L1 or L2 norm). |
| `MaxAbsScaler` | Scales features to the range [-1, 1] by dividing by the maximum absolute value. |
| `MeanVariance` | Standardizes data to have a specified mean and variance (typically mean=0, variance=1). |
| `MinMax` | Scales all values to a range between 0 and 1. |
| `None` | No normalization is applied to the data. |
| `QuantileTransformer` | Transforms features to follow a uniform or normal distribution using quantiles. |
| `RobustScaling` | Scales features using statistics that are robust to outliers. |
| `ZScore` | Standardizes values to have a mean of 0 and a standard deviation of 1. |

