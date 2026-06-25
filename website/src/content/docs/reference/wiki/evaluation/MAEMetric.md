---
title: "MAEMetric<T>"
description: "Computes Mean Absolute Error (MAE): average absolute difference between predictions and actuals."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Evaluation.Metrics.Regression`

Computes Mean Absolute Error (MAE): average absolute difference between predictions and actuals.

## For Beginners

MAE tells you the average magnitude of errors in your predictions.
If MAE = 5, predictions are off by 5 units on average. Unlike MSE, it doesn't penalize
large errors more heavily, making it robust to outliers.

## How It Works

MAE = (1/N) * Σ|y_i - ŷ_i|

