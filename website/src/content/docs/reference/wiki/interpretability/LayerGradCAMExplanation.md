---
title: "LayerGradCAMExplanation<T>"
description: "GradCAM explanation result."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Interpretability.Explainers`

GradCAM explanation result.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `LayerGradCAMExplanation(Vector<>,Matrix<>,Tensor<>,Int32,,Int32,Int32,Int32[])` | Initializes a new LayerGradCAM explanation. |

## Properties

| Property | Summary |
|:-----|:--------|
| `GradCAMMap` | Gets the GradCAM activation map. |
| `Input` | Gets the original input. |
| `InputShape` | Gets the input shape. |
| `LayerHeight` | Gets the layer height. |
| `LayerWidth` | Gets the layer width. |
| `Prediction` | Gets the prediction score. |
| `TargetClass` | Gets the target class. |
| `UpsampledMap` | Gets the upsampled map (if available). |

## Methods

| Method | Summary |
|:-----|:--------|
| `GetNormalizedMap` | Gets normalized GradCAM map (0-1 range). |
| `GetTopRegions(Int32)` | Gets top activated regions. |
| `ToString` | Returns string representation. |

