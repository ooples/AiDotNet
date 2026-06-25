---
title: "GlobalFeatureAblationResult<T>"
description: "Global feature ablation result across a dataset."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Interpretability.Explainers`

Global feature ablation result across a dataset.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `GlobalFeatureAblationResult(Int32[][],String[],Vector<>,Vector<>,Vector<>,Int32)` | Initializes a new global feature ablation result. |

## Properties

| Property | Summary |
|:-----|:--------|
| `FeatureGroups` | Gets the feature groups. |
| `FeatureNames` | Gets feature/group names. |
| `MeanAbsoluteAttributions` | Gets mean absolute attributions (importance magnitude). |
| `MeanAttributions` | Gets mean attributions (can be positive or negative). |
| `NumSamples` | Gets the number of samples analyzed. |
| `StdAttributions` | Gets standard deviation of attributions. |

## Methods

| Method | Summary |
|:-----|:--------|
| `GetSortedFeatures` | Gets features sorted by importance. |
| `GetTopFeatures(Int32)` | Gets the top K most important features globally. |
| `ToString` | Returns a human-readable summary. |

