---
title: "AdditiveDecompositionAlgorithmType"
description: "Represents different algorithm types for additive decomposition of time series data."
section: "API Reference"
---

`Enums` · `AiDotNet.Enums.AlgorithmTypes`

Represents different algorithm types for additive decomposition of time series data.

## For Beginners

Additive decomposition is a technique used to break down time series data 
(data collected over time, like daily temperatures or monthly sales) into separate components:

1. Trend - The long-term direction of the data (going up, down, or staying flat over time)
2. Seasonality - Regular patterns that repeat at fixed intervals (like higher sales during holidays)
3. Residual - The random fluctuations left after accounting for trend and seasonality

It's called "additive" because we assume these components add together to form the original data:
Original Data = Trend + Seasonality + Residual

This enum lists different algorithms that can perform this decomposition, each with its own
approach to separating these components.

## Fields

| Field | Summary |
|:-----|:--------|
| `ExponentialSmoothing` | Uses exponential smoothing to decompose time series data, giving more weight to recent observations. |
| `MovingAverage` | Uses a moving average approach to decompose time series data. |
| `STL` | Uses Seasonal and Trend decomposition using Loess (STL) to break down time series data. |

