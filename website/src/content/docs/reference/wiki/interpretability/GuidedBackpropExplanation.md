---
title: "GuidedBackpropExplanation<T>"
description: "Result of Guided Backpropagation explanation."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Interpretability.Explainers`

Result of Guided Backpropagation explanation.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `GuidedBackpropExplanation(Vector<>,Vector<>,Int32,,Int32[],Tensor<>)` | Initializes a new explanation. |

## Properties

| Property | Summary |
|:-----|:--------|
| `GradientTensor` | Gets the gradient tensor (for tensor inputs). |
| `GuidedGradients` | Gets the guided gradients (attributions). |
| `Input` | Gets the original input. |
| `InputShape` | Gets the input shape (for tensor inputs). |
| `Prediction` | Gets the prediction for the target class. |
| `TargetClass` | Gets the target class that was explained. |

## Methods

| Method | Summary |
|:-----|:--------|
| `GetNormalizedGradients` | Normalizes gradients to [0, 1] range. |
| `GetTopFeatures(Int32)` | Gets the most important input features. |
| `ToString` | Returns string representation. |

