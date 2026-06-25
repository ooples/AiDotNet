---
title: "GuidedGradCAMExplanation<T>"
description: "Guided GradCAM explanation result."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Interpretability.Explainers`

Guided GradCAM explanation result.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `GuidedGradCAMExplanation(Vector<>,Vector<>,Vector<>,Matrix<>,Tensor<>,Int32,,Int32[])` | Initializes a new Guided GradCAM explanation. |

## Properties

| Property | Summary |
|:-----|:--------|
| `GradCAM` | Gets the GradCAM component. |
| `GuidedBackprop` | Gets the Guided Backprop component. |
| `GuidedGradCAM` | Gets the Guided GradCAM result (main output). |
| `Input` | Gets the original input. |
| `InputShape` | Gets the input shape. |
| `Prediction` | Gets the prediction score. |
| `TargetClass` | Gets the target class. |
| `UpsampledGradCAM` | Gets the upsampled GradCAM. |

## Methods

| Method | Summary |
|:-----|:--------|
| `GetGuidedGradCAMTensor` | Gets the Guided GradCAM reshaped to input shape. |
| `GetNormalizedGuidedGradCAM` | Gets normalized Guided GradCAM (0-1 range). |
| `GetTopFeatures(Int32)` | Gets top activated features. |
| `ToString` | Returns string representation. |

