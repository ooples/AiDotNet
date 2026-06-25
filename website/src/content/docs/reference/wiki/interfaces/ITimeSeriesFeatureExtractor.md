---
title: "ITimeSeriesFeatureExtractor<T>"
description: "Defines a specialized data transformer for extracting features from time series data."
section: "API Reference"
---

`Interfaces` · `AiDotNet.Interfaces`

Defines a specialized data transformer for extracting features from time series data.

## For Beginners

Time series data is data collected over time, like:

- Stock prices recorded every minute
- Temperature readings every hour
- Sales figures every day

This interface helps you extract useful features from such data, like:

- Rolling averages (what's the average of the last 7 days?)
- Lagged values (what was the value 3 days ago?)
- Volatility (how much does the value fluctuate?)

These features help machine learning models understand patterns in time series data.

## How It Works

This interface extends the standard data transformer pattern with time series-specific
functionality, including auto-detection of optimal parameters, window-based feature
extraction, and support for both univariate and multivariate time series.

## Properties

| Property | Summary |
|:-----|:--------|
| `AutoDetectEnabled` | Gets whether auto-detection of optimal parameters is enabled. |
| `FeatureNames` | Gets the names of features that will be generated. |
| `InputFeatureCount` | Gets the number of input features (columns) expected. |
| `OutputFeatureCount` | Gets the number of output features that will be generated. |
| `SupportsIncrementalTransform` | Gets whether this transformer supports incremental (streaming) transformation. |
| `WindowSizes` | Gets the window sizes used for rolling calculations. |

## Methods

| Method | Summary |
|:-----|:--------|
| `DetectOptimalWindowSizes(Tensor<>)` | Detects optimal window sizes based on the data's characteristics. |
| `GetIncrementalState` | Gets the current state of the incremental buffer for inspection or serialization. |
| `GetValidationErrors(Tensor<>)` | Gets validation errors for the input data. |
| `InitializeIncremental(Tensor<>)` | Initializes the incremental state from historical data. |
| `SetIncrementalState(IncrementalState<>)` | Restores the incremental state from a previously saved state. |
| `TransformIncremental([])` | Transforms a single new data point incrementally, maintaining internal state. |
| `ValidateInput(Tensor<>)` | Validates that the input data meets the requirements for this transformer. |

