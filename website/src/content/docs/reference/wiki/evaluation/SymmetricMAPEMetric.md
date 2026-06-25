---
title: "SymmetricMAPEMetric<T>"
description: "Computes Symmetric Mean Absolute Percentage Error (sMAPE) for regression."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Evaluation.Metrics.Regression`

Computes Symmetric Mean Absolute Percentage Error (sMAPE) for regression.

## For Beginners

Unlike MAPE, sMAPE treats over-predictions and under-predictions symmetrically:

- Bounded between 0% and 200%
- Symmetric: equal penalty for over and under predictions
- Handles zero values better than MAPE

Note: This is the regression version, not time-series specific.

## How It Works

sMAPE = (100/N) * Σ |y - ŷ| / ((|y| + |ŷ|) / 2)

