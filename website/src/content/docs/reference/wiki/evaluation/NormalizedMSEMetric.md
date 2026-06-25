---
title: "NormalizedMSEMetric<T>"
description: "Computes Normalized Mean Squared Error (NMSE): MSE divided by variance of actuals."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Evaluation.Metrics.Regression`

Computes Normalized Mean Squared Error (NMSE): MSE divided by variance of actuals.

## For Beginners

NMSE provides a scale-independent error measure:

- NMSE = 0: Perfect predictions
- NMSE = 1: Predictions as good as predicting the mean
- NMSE > 1: Worse than predicting the mean
- Equivalent to 1 - R² when computed on the same data

## How It Works

NMSE = MSE / Var(y) = Σ(y - ŷ)² / Σ(y - ȳ)²

