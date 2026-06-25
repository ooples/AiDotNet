---
title: "IForecastingModel<T>"
description: "Interface for time series forecasting models in the Finance module."
section: "API Reference"
---

`Interfaces` · `AiDotNet.Finance.Interfaces`

Interface for time series forecasting models in the Finance module.

## For Beginners

Forecasting models predict future values based on historical patterns.

Key concepts:

- **Lookback window:** How far back in time the model looks to make predictions.
- **Prediction horizon:** How far into the future the model predicts.
- **Univariate:** Forecasting a single variable (e.g., stock price).
- **Multivariate:** Forecasting using multiple variables (e.g., price, volume, indicators).

Example use cases:

- Stock price prediction
- Sales forecasting
- Demand planning
- Energy load forecasting
- Cryptocurrency price prediction

## How It Works

This interface extends `IFinancialModel` with capabilities specific to
time series forecasting, including multi-step prediction, lookback configuration,
and support for both univariate and multivariate forecasting.

## Properties

| Property | Summary |
|:-----|:--------|
| `IsChannelIndependent` | Gets whether this model supports channel-independent (CI) forecasting. |
| `PatchSize` | Gets the patch size for patch-based models (like PatchTST). |
| `Stride` | Gets the stride between consecutive patches. |

## Methods

| Method | Summary |
|:-----|:--------|
| `ApplyInstanceNormalization(Tensor<>)` | Applies instance normalization during inference for distribution shift handling. |
| `AutoregressiveForecast(Tensor<>,Int32)` | Generates multi-step forecasts iteratively (autoregressive forecasting). |
| `Evaluate(Tensor<>,Tensor<>)` | Evaluates the model's forecasting performance on test data. |

