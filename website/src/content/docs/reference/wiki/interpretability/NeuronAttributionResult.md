---
title: "NeuronAttributionResult<T>"
description: "Result of neuron attribution."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Interpretability.Explainers`

Result of neuron attribution.

## Properties

| Property | Summary |
|:-----|:--------|
| `Activations` | Gets or sets the neuron activations. |
| `Attributions` | Gets or sets attribution scores for each neuron. |
| `Instance` | Gets or sets the input instance. |
| `Method` | Gets or sets the attribution method used. |
| `NeuronNames` | Gets or sets neuron names. |
| `Prediction` | Gets or sets the model prediction. |
| `TargetClass` | Gets or sets the target class. |

## Methods

| Method | Summary |
|:-----|:--------|
| `GetTopNeurons(Int32)` | Gets the top K neurons by attribution magnitude. |
| `ToString` | Returns a human-readable summary. |

