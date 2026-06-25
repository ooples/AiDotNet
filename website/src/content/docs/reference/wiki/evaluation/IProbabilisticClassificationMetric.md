---
title: "IProbabilisticClassificationMetric<T>"
description: "Interface for classification metrics that use probabilities."
section: "API Reference"
---

`Interfaces` · `AiDotNet.Evaluation.Metrics`

Interface for classification metrics that use probabilities.

## Methods

| Method | Summary |
|:-----|:--------|
| `Compute(ReadOnlySpan<>,ReadOnlySpan<>,Int32)` | Computes the metric from predicted probabilities and actual labels. |
| `ComputeWithCI(ReadOnlySpan<>,ReadOnlySpan<>,Int32,ConfidenceIntervalMethod,Double,Int32,Nullable<Int32>)` | Computes the metric with confidence interval. |

