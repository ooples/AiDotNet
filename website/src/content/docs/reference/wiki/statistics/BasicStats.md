---
title: "BasicStats<T>"
description: "Provides a collection of basic statistical measures for a set of numeric values."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Statistics`

Provides a collection of basic statistical measures for a set of numeric values.

## For Beginners

Think of BasicStats as a calculator that analyzes a set of numbers and tells you
their important patterns and characteristics.

It answers questions like:

- What's the typical value? (Mean, Median)
- How spread out are the values? (Variance, StandardDeviation)
- What's the range of values? (Min, Max, InterquartileRange)
- Is the distribution skewed or symmetric? (Skewness)
- Are there unusual extreme values? (Kurtosis)

For example, if you have test scores from a class:

- Mean tells you the average score
- StandardDeviation tells you how much scores vary from that average
- Skewness might reveal if more students scored above or below average

These statistics help you understand your data at a glance without having to examine every value.

## How It Works

BasicStats calculates and stores a comprehensive set of descriptive statistics for a collection of values,
including measures of central tendency, dispersion, and distribution shape. These statistics provide
insights into the characteristics of the data distribution.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `BasicStats(BasicStatsInputs<>)` | Initializes a new instance of the BasicStats class with the provided input values. |

## Methods

| Method | Summary |
|:-----|:--------|
| `CalculateStats(Vector<>)` | Calculates all statistical measures from the provided values. |
| `Empty` | Creates an empty BasicStats object with all statistics set to their default values. |
| `GetMetric(MetricType)` | Gets the value of a specific metric based on the provided MetricType. |
| `HasMetric(MetricType)` | Checks if a specific metric is available in this BasicStats instance. |

## Fields

| Field | Summary |
|:-----|:--------|
| `_firstQuartile` | Gets the first quartile (25th percentile) of the dataset. |
| `_interquartileRange` | Gets the interquartile range (IQR) of the dataset. |
| `_mAD` | Gets the median absolute deviation (MAD) of the dataset. |
| `_max` | Gets the maximum value in the dataset. |
| `_mean` | Gets the arithmetic mean (average) of the values. |
| `_median` | Gets the median value of the dataset. |
| `_min` | Gets the minimum value in the dataset. |
| `_n` | Gets the number of values in the dataset. |
| `_numOps` | The numeric operations appropriate for the generic type T. |
| `_standardDeviation` | Gets the standard deviation of the values, a measure of dispersion. |
| `_thirdQuartile` | Gets the third quartile (75th percentile) of the dataset. |
| `_variance` | Gets the variance of the values, a measure of dispersion. |

