---
title: "IMetric<T>"
description: "Base interface for all evaluation metrics."
section: "API Reference"
---

`Interfaces` · `AiDotNet.Evaluation.Metrics`

Base interface for all evaluation metrics.

## Properties

| Property | Summary |
|:-----|:--------|
| `Category` | Gets the category of the metric (Classification, Regression, etc.). |
| `Description` | Gets a human-readable description of the metric. |
| `Direction` | Gets whether higher values indicate better performance. |
| `MaxValue` | Gets the maximum possible value for this metric, if bounded. |
| `MinValue` | Gets the minimum possible value for this metric, if bounded. |
| `Name` | Gets the unique name of the metric. |
| `RequiresProbabilities` | Gets whether this metric requires probability predictions (not just class labels). |
| `SupportsMultiClass` | Gets whether this metric is suitable for multi-class classification. |

