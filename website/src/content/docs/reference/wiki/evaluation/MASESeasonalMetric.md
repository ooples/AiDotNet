---
title: "MASESeasonalMetric<T>"
description: "Computes Seasonal MASE: Mean Absolute Scaled Error with explicit seasonal comparison."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Evaluation.Metrics.TimeSeries`

Computes Seasonal MASE: Mean Absolute Scaled Error with explicit seasonal comparison.

## For Beginners

This is MASE specifically designed for seasonal data:

- Compares your model to a naive seasonal forecast (same period last year/month/week)
- MASE < 1: Model beats seasonal baseline
- Seasonal period could be 7 (daily with weekly pattern), 12 (monthly with yearly pattern), etc.

Essential for evaluating retail, energy, or any data with seasonal patterns.

## How It Works

Seasonal MASE = MAE / MAE_seasonal_naive where seasonal naive predicts y[t-m] (m = seasonal period)

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `MASESeasonalMetric(Int32)` | Initializes the Seasonal MASE metric. |

