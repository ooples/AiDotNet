---
title: "RollingStatsTransformer<T>"
description: "Computes rolling statistics over time series data for feature engineering."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Preprocessing.TimeSeries`

Computes rolling statistics over time series data for feature engineering.

## For Beginners

Rolling statistics are like taking a moving snapshot of your data.

For example, with a 7-day rolling mean:

- Day 7: Average of days 1-7
- Day 8: Average of days 2-8
- Day 9: Average of days 3-9

This helps capture trends and patterns at different time scales. Common uses include:

- Smoothing noisy data (rolling mean)
- Detecting volatility changes (rolling std deviation)
- Finding extreme values (rolling min/max)
- Understanding distribution changes (rolling skewness/kurtosis)

## How It Works

This transformer calculates various statistical measures over rolling windows,
creating features that capture local patterns in time series data.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `RollingStatsTransformer(TimeSeriesFeatureOptions)` | Creates a new rolling statistics transformer with the specified options. |

## Properties

| Property | Summary |
|:-----|:--------|
| `SupportsInverseTransform` |  |

## Methods

| Method | Summary |
|:-----|:--------|
| `BuildOperationNames` | Builds the list of operation names based on enabled statistics. |
| `ComputeIQR(Double[])` | Computes the interquartile range (Q3 - Q1). |
| `ComputeIncrementalFeatures(IncrementalState<>,[])` | Computes rolling statistics features incrementally from the circular buffer. |
| `ComputeKurtosis(Double[])` | Computes the excess kurtosis (tail heaviness measure). |
| `ComputeMAD(Double[])` | Computes the median absolute deviation. |
| `ComputeMedian(Double[])` | Computes the median of the values. |
| `ComputePercentile(Double[],Double)` | Computes a percentile value. |
| `ComputeSkewness(Double[])` | Computes the skewness (asymmetry measure). |
| `ComputeStatistics(Double[],Tensor<>,Int32,Int32)` | Computes all enabled statistics for a window and writes to output. |
| `ComputeStatisticsIncremental(Double[],[],Int32)` | Computes all enabled statistics for a window and writes to the features array. |
| `ComputeStatisticsToArray(Double[],Tensor<>,Int32,Int32)` | Thread-safe version that computes stats starting at a specific output index. |
| `ComputeStdDev(Double[])` | Computes the sample standard deviation. |
| `ComputeVariance(Double[])` | Computes the sample variance. |
| `CountEnabledStats` | Counts the number of enabled statistics. |
| `ExportParameters` | Exports transformer-specific parameters for serialization. |
| `ExtractWindow(Tensor<>,Int32,Int32,Int32)` | Extracts data for a rolling window ending at the specified time step. |
| `ExtractWindowWithSize(Tensor<>,Int32,Int32,Int32,Int32)` | Extracts data for a rolling window with support for partial windows. |
| `FitCore(Tensor<>)` |  |
| `GenerateFeatureNames` |  |
| `GetOperationNames` |  |
| `ImportParameters(Dictionary<String,Object>)` | Imports transformer-specific parameters. |
| `TransformCore(Tensor<>)` |  |
| `TransformParallel(Tensor<>)` |  |

## Fields

| Field | Summary |
|:-----|:--------|
| `_enabledStats` | The enabled statistics from options. |
| `_operationNames` | Cached operation names for feature naming (readonly for thread safety). |
| `_percentiles` | Custom percentiles to compute. |

