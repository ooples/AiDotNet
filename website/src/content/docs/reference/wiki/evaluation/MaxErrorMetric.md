---
title: "MaxErrorMetric<T>"
description: "Computes Maximum Error: the worst-case absolute prediction error."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Evaluation.Metrics.Regression`

Computes Maximum Error: the worst-case absolute prediction error.

## For Beginners

Maximum error shows you the worst single prediction your model made.
This is important for applications where even one large error is unacceptable (safety-critical systems,
financial predictions with risk limits). Unlike MAE or RMSE, it's not influenced by average performance

- it only cares about the worst case.

## How It Works

Max Error = max(|y_i - ŷ_i|)

