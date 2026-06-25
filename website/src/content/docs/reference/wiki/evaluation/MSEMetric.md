---
title: "MSEMetric<T>"
description: "Computes Mean Squared Error (MSE): average squared difference between predictions and actuals."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Evaluation.Metrics.Regression`

Computes Mean Squared Error (MSE): average squared difference between predictions and actuals.

## For Beginners

MSE penalizes large errors more heavily than small ones due to squaring.
Good for optimizing models (differentiable), but harder to interpret than MAE.

## How It Works

MSE = (1/N) * Σ(y_i - ŷ_i)²

