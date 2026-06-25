---
title: "RelativeSquaredErrorMetric<T>"
description: "Computes Relative Squared Error (RSE): sum of squared errors relative to baseline model."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Evaluation.Metrics.Regression`

Computes Relative Squared Error (RSE): sum of squared errors relative to baseline model.

## For Beginners

RSE compares your model to a baseline (mean predictor):

- RSE < 1: Your model is better than the mean baseline
- RSE = 1: Your model is equivalent to predicting the mean
- RSE > 1: Your model is worse than the mean baseline
- RSE = 1 - R² (when calculated correctly)

## How It Works

RSE = Σ(y - ŷ)² / Σ(y - ȳ)²

