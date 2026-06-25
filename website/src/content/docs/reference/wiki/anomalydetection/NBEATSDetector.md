---
title: "NBEATSDetector<T>"
description: "Implements N-BEATS (Neural Basis Expansion Analysis for Time Series) for anomaly detection."
section: "API Reference"
---

`Models & Types` · `AiDotNet.AnomalyDetection.TimeSeries`

Implements N-BEATS (Neural Basis Expansion Analysis for Time Series) for anomaly detection.

## For Beginners

N-BEATS is a deep neural architecture for time series forecasting
that uses stacked blocks with basis expansion. For anomaly detection, it predicts
the next value and uses the prediction error as the anomaly score.

## How It Works

The algorithm works by:

1. Stack multiple blocks, each outputting a partial forecast and backcast
2. Each block uses fully-connected layers with basis expansion
3. Residual learning allows progressive refinement
4. High prediction errors indicate anomalies

**When to use:**

- Univariate or multivariate time series
- When interpretability of forecasts matters
- Long-horizon forecasting-based anomaly detection

**Industry Standard Defaults:**

- Number of stacks: 2
- Number of blocks per stack: 3
- Hidden dimensions: 64
- Lookback: 10
- Epochs: 50
- Contamination: 0.1 (10%)

Reference: Oreshkin, B. N., et al. (2020).
"N-BEATS: Neural Basis Expansion Analysis for Interpretable Time Series Forecasting." ICLR.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `NBEATSDetector(Int32,Int32,Int32,Int32,Int32,Double,Double,Int32)` | Creates a new N-BEATS anomaly detector. |

## Properties

| Property | Summary |
|:-----|:--------|
| `Lookback` | Gets the lookback window size. |
| `NumBlocks` | Gets the number of blocks per stack. |
| `NumStacks` | Gets the number of stacks. |

## Methods

| Method | Summary |
|:-----|:--------|
| `Fit(Matrix<>)` |  |
| `ScoreAnomalies(Matrix<>)` |  |

