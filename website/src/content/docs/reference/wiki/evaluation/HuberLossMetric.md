---
title: "HuberLossMetric<T>"
description: "Computes Huber Loss: a robust loss function that is less sensitive to outliers than MSE."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Evaluation.Metrics.Regression`

Computes Huber Loss: a robust loss function that is less sensitive to outliers than MSE.

## For Beginners

Huber loss combines the best of MSE and MAE:

- For small errors (≤ δ): Uses squared error like MSE (smooth gradient)
- For large errors (> δ): Uses linear error like MAE (robust to outliers)

The delta parameter controls where the transition happens. Common values are 1.0 or 1.35.

## How It Works

Huber Loss = (1/N) * Σ L_δ(y, ŷ) where:

- L_δ = 0.5 * (y - ŷ)² if |y - ŷ| ≤ δ (quadratic for small errors)
- L_δ = δ * (|y - ŷ| - 0.5 * δ) if |y - ŷ| > δ (linear for large errors)

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `HuberLossMetric(Double)` | Initializes the Huber Loss metric. |

