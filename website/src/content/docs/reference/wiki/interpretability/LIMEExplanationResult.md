---
title: "LIMEExplanationResult<T>"
description: "Represents a LIME explanation result."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Interpretability.Explainers`

Represents a LIME explanation result.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `LIMEExplanationResult(Vector<>,,,,String[])` | Initializes a new LIME explanation result. |

## Properties

| Property | Summary |
|:-----|:--------|
| `Coefficients` | Gets the coefficients of the local linear model (feature weights). |
| `FeatureNames` | Gets the feature names, if available. |
| `Intercept` | Gets the intercept of the local linear model. |
| `LocalR2` | Gets the R² score of the local linear model (how well it fits locally). |
| `NumFeatures` | Gets the number of features. |
| `Prediction` | Gets the original prediction being explained. |

## Methods

| Method | Summary |
|:-----|:--------|
| `GetSortedFeatures` | Gets features sorted by absolute coefficient (most important first). |
| `GetTopFeatures(Int32)` | Gets the top N most important features. |
| `ToString` | Returns a human-readable summary. |

