---
title: "OcclusionExplanation<T>"
description: "Result of occlusion analysis."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Interpretability.Explainers`

Result of occlusion analysis.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `OcclusionExplanation(Tensor<>,Tensor<>,Int32,,Int32[],Int32[])` | Initializes a new occlusion explanation. |

## Properties

| Property | Summary |
|:-----|:--------|
| `BasePrediction` | Gets the base prediction for the target class. |
| `Input` | Gets the original input. |
| `SensitivityMap` | Gets the sensitivity map showing importance of each region. |
| `Strides` | Gets the strides used for sliding. |
| `TargetClass` | Gets the target class that was explained. |
| `WindowShape` | Gets the window shape used for occlusion. |

## Methods

| Method | Summary |
|:-----|:--------|
| `GetMostImportantRegions(Int32)` | Gets the most important regions (highest sensitivity). |
| `GetUpsampledMap` | Upsamples the sensitivity map to match input size. |
| `ToString` | Returns a human-readable summary. |

