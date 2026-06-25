---
title: "MeanAbsolutePercentageErrorMetric<T>"
description: "Computes Mean Absolute Percentage Error with configurable handling of zeros."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Evaluation.Metrics.Regression`

Computes Mean Absolute Percentage Error with configurable handling of zeros.

## For Beginners

MAPE expresses error as a percentage:

- Intuitive interpretation (e.g., "10% average error")
- Scale-independent
- Warning: undefined when actual values are zero
- Asymmetric: penalizes over-predictions less than under-predictions

## How It Works

MAPE = (100/N) × Σ |y - ŷ| / |y|

