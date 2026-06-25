---
title: "TheilUMetric<T>"
description: "Computes Theil's U Statistic: measures forecast accuracy relative to a naive no-change forecast."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Evaluation.Metrics.TimeSeries`

Computes Theil's U Statistic: measures forecast accuracy relative to a naive no-change forecast.

## For Beginners

Theil's U compares your model to the simplest possible forecast (no change):

- U = 0: Perfect predictions
- U = 1: Model is as accurate as naive "no change" forecast
- U < 1: Model outperforms naive forecast
- U > 1: Model is worse than simply predicting no change

Particularly useful for economic forecasting.

## How It Works

Theil's U = √[Σ(ŷ_t - y_t)² / Σ(y_t - y_{t-1})²]

