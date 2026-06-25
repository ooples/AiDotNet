---
title: "QuantileForecastResult<T>"
description: "Represents the result of a probabilistic/quantile forecast from a time series foundation model."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Finance.Forecasting`

Represents the result of a probabilistic/quantile forecast from a time series foundation model.

## For Beginners

A quantile forecast doesn't just predict a single value — it predicts
a range of possible values with associated probabilities. For example:

- 10th percentile: There's a 10% chance the actual value will be below this
- 50th percentile (median): The "middle" forecast — equally likely to be above or below
- 90th percentile: There's a 90% chance the actual value will be below this

This gives you prediction intervals (confidence bands) instead of just point predictions.

## How It Works

**Supported Models:** Chronos-2, Moirai 2.0, TimesFM 2.5, Chronos-Bolt, Lag-Llama,
and diffusion-based models (TimeGrad, CSDI, TSDiff) that produce sample-based forecasts.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `QuantileForecastResult(IReadOnlyList<Tensor<>>,Double[])` | Creates a quantile forecast result from sample trajectories. |
| `QuantileForecastResult(Tensor<>,Dictionary<Double,Tensor<>>)` | Creates a quantile forecast result from precomputed quantile tensors. |

## Properties

| Property | Summary |
|:-----|:--------|
| `Horizon` | Gets the forecast horizon (number of predicted time steps). |
| `PointForecast` | Gets the point forecast (median or mean prediction) of shape [horizon]. |
| `QuantileForecasts` | Gets the quantile forecasts as a dictionary mapping quantile level to forecast tensor. |
| `QuantileLevels` | Gets the quantile levels (e.g., [0.1, 0.25, 0.5, 0.75, 0.9]). |

## Methods

| Method | Summary |
|:-----|:--------|
| `GetPredictionInterval(Double)` | Gets the prediction interval at the specified confidence level. |

