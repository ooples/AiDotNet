---
title: "LRPExplanation<T>"
description: "Represents the result of an LRP analysis."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Interpretability.Explainers`

Represents the result of an LRP analysis.

## Properties

| Property | Summary |
|:-----|:--------|
| `FeatureNames` | Gets or sets the feature names. |
| `Input` | Gets or sets the input instance. |
| `OutputIndex` | Gets or sets the output index that was explained. |
| `OutputValue` | Gets or sets the output value being explained. |
| `Prediction` | Gets or sets the model prediction. |
| `RelevanceScores` | Gets or sets the relevance scores for each input feature. |
| `Rule` | Gets or sets the LRP rule used. |
| `TotalNegativeRelevance` | Gets or sets the sum of negative relevance scores. |
| `TotalPositiveRelevance` | Gets or sets the sum of positive relevance scores. |

## Methods

| Method | Summary |
|:-----|:--------|
| `GetNegativeRelevance` | Gets features with negative relevance (contradicting the prediction). |
| `GetPositiveRelevance` | Gets features with positive relevance (supporting the prediction). |
| `GetSortedRelevance` | Gets relevance scores sorted by absolute value (most relevant first). |
| `ToString` | Returns a human-readable summary. |

