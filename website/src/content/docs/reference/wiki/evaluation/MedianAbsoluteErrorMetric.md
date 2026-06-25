---
title: "MedianAbsoluteErrorMetric<T>"
description: "Computes Median Absolute Error (MedAE): the median of all absolute errors."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Evaluation.Metrics.Regression`

Computes Median Absolute Error (MedAE): the median of all absolute errors.

## For Beginners

Median Absolute Error is even more robust to outliers than MAE:

- Uses the median instead of mean, so outliers have minimal impact
- Represents the "typical" error - half your predictions are better, half are worse
- Particularly useful when your data has heavy-tailed error distributions

Compare MedAE vs MAE: if MedAE is much smaller than MAE, you have outlier issues.

## How It Works

MedAE = median(|y_i - ŷ_i|)

