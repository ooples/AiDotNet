---
title: "RMSEMetric<T>"
description: "Computes Root Mean Squared Error (RMSE): square root of MSE."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Evaluation.Metrics.Regression`

Computes Root Mean Squared Error (RMSE): square root of MSE.

## For Beginners

RMSE is MSE in the same units as the target variable,
making it more interpretable. If RMSE = 5 for house prices in $1000s, errors average ~$5000.

## How It Works

RMSE = √MSE = √[(1/N) * Σ(y_i - ŷ_i)²]

