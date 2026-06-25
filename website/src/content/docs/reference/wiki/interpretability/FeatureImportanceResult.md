---
title: "FeatureImportanceResult<T>"
description: "Represents the result of permutation feature importance calculation."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Interpretability.Explainers`

Represents the result of permutation feature importance calculation.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `FeatureImportanceResult(Vector<>,Vector<>,,String[])` | Initializes a new feature importance result. |

## Properties

| Property | Summary |
|:-----|:--------|
| `BaselineScore` | Gets the baseline model score (before any permutation). |
| `FeatureNames` | Gets the feature names, if available. |
| `ImportanceStds` | Gets the standard deviation of importance scores across repeats. |
| `Importances` | Gets the importance score for each feature (mean drop in performance when permuted). |
| `NumFeatures` | Gets the number of features. |

## Methods

| Method | Summary |
|:-----|:--------|
| `GetSortedFeatures` | Gets features sorted by importance (most important first). |
| `GetTopFeatures(Int32)` | Gets the top N most important features. |
| `ToDictionary` | Converts to a dictionary mapping feature names/indices to importance scores. |
| `ToString` | Returns a human-readable summary. |

