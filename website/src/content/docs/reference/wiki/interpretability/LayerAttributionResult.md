---
title: "LayerAttributionResult<T>"
description: "Result of layer attribution."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Interpretability.Explainers`

Result of layer attribution.

## Properties

| Property | Summary |
|:-----|:--------|
| `Activations` | Gets or sets the layer activations. |
| `Attributions` | Gets or sets attribution scores for each neuron in the layer. |
| `Instance` | Gets or sets the input instance. |
| `Method` | Gets or sets the attribution method used. |
| `Prediction` | Gets or sets the model prediction. |
| `TargetClass` | Gets or sets the target class. |

## Methods

| Method | Summary |
|:-----|:--------|
| `GetAttributionSum` | Gets the sum of attributions. |
| `GetChannelAttributions(Int32,Int32,Int32)` | Gets channel-wise attribution sums (for conv layers). |
| `GetTopNeurons(Int32)` | Gets the top K neurons by attribution magnitude. |
| `ReshapeSpatial(Int32,Int32,Int32)` | Reshapes attributions for spatial layers (e.g., conv layers). |
| `ToString` | Returns a human-readable summary. |

