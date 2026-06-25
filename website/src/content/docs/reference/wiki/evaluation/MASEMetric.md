---
title: "MASEMetric<T>"
description: "Computes Mean Absolute Scaled Error (MASE): scale-independent measure for time series."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Evaluation.Metrics.TimeSeries`

Computes Mean Absolute Scaled Error (MASE): scale-independent measure for time series.

## For Beginners

MASE compares your model to a simple baseline (naive forecast).

- MASE < 1: Model beats the naive baseline
- MASE = 1: Model is as good as naive
- MASE > 1: Model is worse than naive

## How It Works

MASE = MAE / MAE_naive where MAE_naive is the MAE of a naive seasonal forecast.

