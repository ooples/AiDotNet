---
title: "MeanDirectionalAccuracyMetric<T>"
description: "Computes Mean Directional Accuracy (MDA): fraction of correctly predicted directions."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Evaluation.Metrics.Regression`

Computes Mean Directional Accuracy (MDA): fraction of correctly predicted directions.

## For Beginners

MDA measures if you predicted the right direction:

- Did the model predict "up" when actual went up?
- Range: 0 to 1, higher is better
- 0.5 = random guessing
- Important in trading/forecasting where direction matters more than magnitude

## How It Works

MDA = (1/N) × Σ I(sign(y_t - y_{t-1}) = sign(ŷ_t - ŷ_{t-1}))

