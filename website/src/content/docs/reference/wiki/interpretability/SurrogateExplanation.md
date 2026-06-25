---
title: "SurrogateExplanation<T>"
description: "Represents a global surrogate model explanation."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Interpretability.Explainers`

Represents a global surrogate model explanation.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `SurrogateExplanation(Vector<>,,,String[])` | Initializes a new surrogate explanation. |

## Properties

| Property | Summary |
|:-----|:--------|
| `Coefficients` | Gets the coefficients of the linear surrogate model. |
| `FeatureNames` | Gets the feature names, if available. |
| `Fidelity` | Gets the fidelity (R²) indicating how well the surrogate approximates the black box. |
| `Intercept` | Gets the intercept of the linear surrogate model. |
| `NumFeatures` | Gets the number of features. |

## Methods

| Method | Summary |
|:-----|:--------|
| `GetSortedFeatures` | Gets features sorted by absolute coefficient (most important first). |
| `GetTopFeatures(Int32)` | Gets the top N most important features according to the surrogate model. |
| `ToString` | Returns a human-readable summary. |

