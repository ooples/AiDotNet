---
title: "MeanSquaredLogErrorMetric<T>"
description: "Computes Mean Squared Logarithmic Error (MSLE): squared version of RMSLE."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Evaluation.Metrics.Regression`

Computes Mean Squared Logarithmic Error (MSLE): squared version of RMSLE.

## For Beginners

MSLE is useful when:

- You care about percentage errors
- Under-predictions should be penalized more than over-predictions
- Target values span multiple orders of magnitude

The squared version penalizes large errors more than RMSLE.

## How It Works

MSLE = (1/N) * Σ(log(1 + y_i) - log(1 + ŷ_i))²

