---
title: "FinancialPreprocessor<T>"
description: "Preprocesses financial time series data into model-ready tensors."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Finance.Data`

Preprocesses financial time series data into model-ready tensors.

## For Beginners

This class is the "data prep" step. It turns
messy price bars into clean numeric tensors that models can learn from.
It also supports windowing (lookback) and scaling (normalization).

## How It Works

FinancialPreprocessor converts raw OHLCV data into feature tensors,
builds rolling windows for supervised learning, and provides basic
normalization utilities for time series.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `FinancialPreprocessor(INumericOperations<>)` | Creates a new financial preprocessor. |

## Methods

| Method | Summary |
|:-----|:--------|
| `BuildFeatureVector(IReadOnlyList<MarketDataPoint<>>,Int32,Boolean,Boolean)` | Builds a feature vector for a single time step. |
| `ComputeReturn(IReadOnlyList<MarketDataPoint<>>,Int32)` | Computes the close-to-close return for a given index. |
| `CreateFeatureTensor(IReadOnlyList<MarketDataPoint<>>,Boolean,Boolean)` | Converts raw market data into a 2D feature tensor. |
| `CreateSupervisedLearningTensors(IReadOnlyList<MarketDataPoint<>>,Int32,Int32,Boolean,Boolean,Boolean)` | Creates windowed features and targets for supervised forecasting. |
| `GetFeatureCount(Boolean,Boolean)` | Gets the number of features produced for each time step. |
| `InitializeMinMax(Tensor<>,[],[],Int32)` | Initializes min/max arrays from the first feature row. |
| `NormalizeMinMax(Tensor<>,ValueTuple<Vector<>,Vector<>>)` | Applies min-max normalization across the feature dimension. |
| `NormalizeZScore(Tensor<>,ValueTuple<Vector<>,Vector<>>)` | Applies z-score normalization across the feature dimension. |
| `ValidateSeries(IReadOnlyList<MarketDataPoint<>>,String)` | Validates that the series is non-empty. |
| `ValidateWindowSizes(Int32,Int32)` | Validates the lookback and horizon settings. |

