---
title: "WAPEMetric<T>"
description: "Computes Weighted Absolute Percentage Error (WAPE): total absolute error as percentage of total actuals."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Evaluation.Metrics.TimeSeries`

Computes Weighted Absolute Percentage Error (WAPE): total absolute error as percentage of total actuals.

## For Beginners

WAPE gives a single number measuring overall forecast accuracy:

- More robust than MAPE when dealing with intermittent demand (zeros)
- Weights errors by the magnitude of actuals
- WAPE = 0.1 means your total errors are 10% of total actual values

Also known as MAD/Mean ratio in some contexts.

## How It Works

WAPE = Σ|y - ŷ| / Σ|y|

