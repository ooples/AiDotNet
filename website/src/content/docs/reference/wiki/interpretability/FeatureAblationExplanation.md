---
title: "FeatureAblationExplanation<T>"
description: "Result of feature ablation for a single input."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Interpretability.Explainers`

Result of feature ablation for a single input.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `FeatureAblationExplanation(Vector<>,Vector<>,Vector<>,Int32[][],String[],Int32,,Vector<>)` | Initializes a new feature ablation explanation. |

## Properties

| Property | Summary |
|:-----|:--------|
| `AblatedPredictions` | Gets predictions after ablating each feature/group. |
| `Attributions` | Gets attributions for each feature/group. |
| `BasePrediction` | Gets the base prediction score. |
| `Baseline` | Gets the baseline values used. |
| `FeatureGroups` | Gets the feature groups. |
| `FeatureNames` | Gets feature/group names. |
| `Input` | Gets the original input. |
| `NumGroups` | Gets the number of feature groups. |
| `TargetClass` | Gets the target class explained. |

## Methods

| Method | Summary |
|:-----|:--------|
| `GetSortedFeatures` | Gets features sorted by importance (absolute attribution). |
| `GetTopFeatures(Int32)` | Gets the top K most important features. |
| `ToString` | Returns a human-readable summary. |

