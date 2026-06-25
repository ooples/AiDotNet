---
title: "RMSLEMetric<T>"
description: "Computes Root Mean Squared Logarithmic Error (RMSLE): measures ratio errors rather than absolute errors."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Evaluation.Metrics.Regression`

Computes Root Mean Squared Logarithmic Error (RMSLE): measures ratio errors rather than absolute errors.

## For Beginners

RMSLE is useful when:

- You care about percentage/ratio errors rather than absolute differences
- Under-predictions should be penalized more than over-predictions
- The target spans several orders of magnitude (like prices or counts)

Note: Requires all values to be non-negative (≥ 0) since we take logarithms.

## How It Works

RMSLE = √[(1/N) * Σ(log(1 + y_i) - log(1 + ŷ_i))²]

