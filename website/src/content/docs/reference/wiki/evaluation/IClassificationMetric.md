---
title: "IClassificationMetric<T>"
description: "Interface for classification metrics."
section: "API Reference"
---

`Interfaces` · `AiDotNet.Evaluation.Metrics`

Interface for classification metrics.

## Methods

| Method | Summary |
|:-----|:--------|
| `Compute(ReadOnlySpan<>,ReadOnlySpan<>)` | Computes the metric from predicted and actual class labels. |
| `ComputeWithCI(ReadOnlySpan<>,ReadOnlySpan<>,ConfidenceIntervalMethod,Double,Int32,Nullable<Int32>)` | Computes the metric with full result including confidence interval. |

