---
title: "IRegressionMetric<T>"
description: "Interface for regression metrics."
section: "API Reference"
---

`Interfaces` · `AiDotNet.Evaluation.Metrics`

Interface for regression metrics.

## Methods

| Method | Summary |
|:-----|:--------|
| `Compute(ReadOnlySpan<>,ReadOnlySpan<>)` | Computes the metric from predicted and actual values. |
| `ComputeWithCI(ReadOnlySpan<>,ReadOnlySpan<>,ConfidenceIntervalMethod,Double,Int32,Nullable<Int32>)` | Computes the metric with confidence interval. |

