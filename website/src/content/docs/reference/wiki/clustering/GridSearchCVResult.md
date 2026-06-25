---
title: "GridSearchCVResult<T>"
description: "Complete result from cross-validation grid search."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Clustering.AutoML`

Complete result from cross-validation grid search.

## Properties

| Property | Summary |
|:-----|:--------|
| `AllTrials` | All successful trials, ranked by mean score. |
| `BestResult` | The best result found. |
| `HigherIsBetter` | Whether higher metric values are better. |
| `NumFolds` | Number of cross-validation folds. |
| `PrimaryMetric` | The metric used for optimization. |
| `SuccessfulTrials` | Number of successful trials. |
| `TotalCombinations` | Total number of parameter combinations tried. |

## Methods

| Method | Summary |
|:-----|:--------|
| `GetSummary` | Gets a summary of the best result. |

