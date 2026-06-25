---
title: "FeatureInteractionResult<T>"
description: "Represents the result of a Feature Interaction analysis."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Interpretability.Explainers`

Represents the result of a Feature Interaction analysis.

## Properties

| Property | Summary |
|:-----|:--------|
| `FeatureNames` | Gets or sets the feature names. |
| `OverallHStatistics` | Gets or sets the overall H-statistics for each feature. |
| `PairwiseHStatistics` | Gets or sets the pairwise H-statistics ((featureI, featureJ) -> H value). |

## Methods

| Method | Summary |
|:-----|:--------|
| `InterpretHStatistic(Double)` | Gets the interpretation of an H-statistic value. |
| `ToString` | Returns a human-readable summary. |

