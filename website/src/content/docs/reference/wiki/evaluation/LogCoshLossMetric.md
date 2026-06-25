---
title: "LogCoshLossMetric<T>"
description: "Computes Log-Cosh Loss: mean of log(cosh(y - ŷ))."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Evaluation.Metrics.Regression`

Computes Log-Cosh Loss: mean of log(cosh(y - ŷ)).

## For Beginners

Log-Cosh loss combines benefits of MSE and MAE:

- Behaves like MAE for large errors (robust to outliers)
- Behaves like MSE for small errors (smooth gradient)
- Always positive and convex
- Has continuous second derivative (unlike Huber)

## How It Works

Log-Cosh = (1/N) × Σ log(cosh(y - ŷ))

**Advantages over alternatives:**

- vs MSE: More robust to outliers
- vs MAE: Smoother gradient near zero
- vs Huber: No hyperparameter to tune

