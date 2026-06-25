---
title: "CrossValidationResult<T>"
description: "Aggregated results from cross-validation across all folds."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Evaluation.Engines`

Aggregated results from cross-validation across all folds.

## Properties

| Property | Summary |
|:-----|:--------|
| `AggregatedMetrics` | Metrics aggregated across all folds with mean, std, and confidence intervals. |
| `FoldResults` | Results from each individual fold. |
| `NumFolds` | Number of folds executed. |
| `StrategyName` | Name of the cross-validation strategy used. |

## Methods

| Method | Summary |
|:-----|:--------|
| `GetMeanMetric(String)` | Gets the mean value for a specific metric. |
| `GetStdMetric(String)` | Gets the standard deviation for a specific metric across folds. |
| `TryGetMeanMetric(String,)` | Tries to get the mean value for a specific metric. |

