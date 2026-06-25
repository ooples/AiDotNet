---
title: "ColdStartAnalysisResult<T>"
description: "Cold start analysis result showing initial sample selection performance."
section: "API Reference"
---

`Models & Types` · `AiDotNet.ActiveLearning.Results`

Cold start analysis result showing initial sample selection performance.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `ColdStartAnalysisResult(INumericOperations<>)` | Initializes a new instance with default values. |

## Properties

| Property | Summary |
|:-----|:--------|
| `ClassDistribution` | Gets or sets the class balance of the initial sample. |
| `InitialAccuracy` | Gets or sets the accuracy after initial training. |
| `InitialSampleCount` | Gets or sets the number of initial samples selected. |
| `Representativeness` | Gets or sets how representative the initial sample is of the full dataset. |
| `SelectionTime` | Gets or sets the time to select initial samples. |
| `StrategyName` | Gets or sets the cold start strategy used. |

