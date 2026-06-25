---
title: "QuantileLossMetric<T>"
description: "Computes Quantile Loss (Pinball Loss) for quantile regression."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Evaluation.Metrics.Regression`

Computes Quantile Loss (Pinball Loss) for quantile regression.

## For Beginners

Quantile loss is used when you want to predict a specific percentile:

- τ = 0.5: Median (equivalent to MAE)
- τ = 0.9: 90th percentile (over-predictions penalized less)
- τ = 0.1: 10th percentile (under-predictions penalized less)

## How It Works

Quantile Loss = (1/N) × Σ max(τ(y - ŷ), (τ - 1)(y - ŷ))

**Use cases:**

- Risk assessment (predict 95th percentile of losses)
- Inventory planning (predict 99th percentile of demand)
- Uncertainty quantification (predict multiple quantiles)

