---
title: "MAPEMetric<T>"
description: "Computes Mean Absolute Percentage Error (MAPE): average percentage error."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Evaluation.Metrics.Regression`

Computes Mean Absolute Percentage Error (MAPE): average percentage error.

## For Beginners

MAPE expresses error as a percentage of actual values.
MAPE = 5% means predictions are off by 5% on average. Easy to interpret but undefined when
actual values are zero.

## How It Works

MAPE = (100/N) * Σ|y_i - ŷ_i| / |y_i|

**Limitations:** Undefined for zero actuals, asymmetric (penalizes over-predictions
less than under-predictions when actuals are large).

