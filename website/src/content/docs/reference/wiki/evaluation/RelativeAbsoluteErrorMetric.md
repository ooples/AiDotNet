---
title: "RelativeAbsoluteErrorMetric<T>"
description: "Computes Relative Absolute Error (RAE): sum of absolute errors relative to baseline."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Evaluation.Metrics.Regression`

Computes Relative Absolute Error (RAE): sum of absolute errors relative to baseline.

## For Beginners

RAE is the absolute error version of RSE:

- RAE < 1: Your model is better than the mean baseline
- RAE = 1: Your model equals the mean baseline
- More robust to outliers than RSE
- Commonly used in time series forecasting

## How It Works

RAE = Σ|y - ŷ| / Σ|y - ȳ|

