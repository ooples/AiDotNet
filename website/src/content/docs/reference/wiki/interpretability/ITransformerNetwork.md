---
title: "ITransformerNetwork<T, TInput, TOutput>"
description: "Interface for transformer networks that support attention visualization."
section: "API Reference"
---

`Interfaces` · `AiDotNet.Interpretability.Interfaces`

Interface for transformer networks that support attention visualization.

## For Beginners

Transformers use "attention" to focus on different parts
of the input when making predictions. This interface provides methods to extract
these attention patterns for visualization and analysis.

## How It Works

For example, in a text classifier, attention might show which words the model
focused on most when making its classification decision.

## Methods

| Method | Summary |
|:-----|:--------|
| `GetAttentionWeights(Tensor<>)` | Gets attention weights from all transformer layers. |

