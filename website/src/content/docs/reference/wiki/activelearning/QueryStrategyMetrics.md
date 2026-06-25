---
title: "QueryStrategyMetrics<T>"
description: "Detailed metrics for query strategy performance analysis."
section: "API Reference"
---

`Models & Types` · `AiDotNet.ActiveLearning.Results`

Detailed metrics for query strategy performance analysis.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `QueryStrategyMetrics(INumericOperations<>)` | Initializes a new instance with default values. |

## Properties

| Property | Summary |
|:-----|:--------|
| `AverageDiversity` | Gets or sets the diversity of selected samples. |
| `AverageInformativeness` | Gets or sets the average informativeness score across all queries. |
| `AverageSelectionTime` | Gets or sets the average time to select a batch. |
| `HasSelectionCorrelation` | Gets or sets whether selection correlation was calculated. |
| `InformativenessVariance` | Gets or sets the variance of informativeness scores. |
| `SelectionCorrelation` | Gets or sets the correlation between selection order and true usefulness. |
| `StrategyName` | Gets or sets the name of the query strategy. |

