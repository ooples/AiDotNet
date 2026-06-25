---
title: "GlobalDeepSHAPExplanation<T>"
description: "Represents global DeepSHAP feature importance."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Interpretability.Explainers`

Represents global DeepSHAP feature importance.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `GlobalDeepSHAPExplanation(DeepSHAPExplanation<>[],String[])` | Initializes a new global DeepSHAP explanation. |

## Properties

| Property | Summary |
|:-----|:--------|
| `FeatureNames` | Gets the feature names. |
| `LocalExplanations` | Gets the local explanations used to compute global importance. |
| `MeanAbsoluteAttributions` | Gets the average absolute attribution for each feature. |

## Methods

| Method | Summary |
|:-----|:--------|
| `GetSortedImportance` | Gets features sorted by global importance. |
| `ToString` | Returns a human-readable summary. |

