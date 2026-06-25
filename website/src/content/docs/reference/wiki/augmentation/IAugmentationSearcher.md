---
title: "IAugmentationSearcher<T, TData>"
description: "Interface for AutoML search algorithms over augmentation spaces."
section: "API Reference"
---

`Interfaces` · `AiDotNet.Augmentation`

Interface for AutoML search algorithms over augmentation spaces.

## Properties

| Property | Summary |
|:-----|:--------|
| `EvaluationCount` | Gets the number of evaluations performed. |
| `SearchSpace` | Gets the search space being explored. |

## Methods

| Method | Summary |
|:-----|:--------|
| `GetBest` | Gets the best configuration found so far. |
| `ReportResult(IList<SampledConfiguration>,Double)` | Reports the result of evaluating a configuration. |
| `SuggestNext` | Suggests the next configuration to evaluate. |

