---
title: "SMAPEMetric<T>"
description: "Computes Symmetric Mean Absolute Percentage Error (SMAPE): a bounded percentage error metric."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Evaluation.Metrics.TimeSeries`

Computes Symmetric Mean Absolute Percentage Error (SMAPE): a bounded percentage error metric.

## For Beginners

SMAPE fixes some issues with MAPE:

- Bounded between 0% and 200% (unlike MAPE which can be infinite)
- Treats over-predictions and under-predictions more symmetrically
- Still handles zero values better than MAPE

Common in forecasting competitions (e.g., M-competitions).

## How It Works

SMAPE = (100/N) * Σ |y - ŷ| / ((|y| + |ŷ|) / 2)

