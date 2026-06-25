---
title: "PoolingStrategy<T>"
description: "Defines the available pooling strategies for creating a single sentence embedding."
section: "API Reference"
---

`Enums` · `AiDotNet.NeuralNetworks`

Defines the available pooling strategies for creating a single sentence embedding.

## Fields

| Field | Summary |
|:-----|:--------|
| `ClsToken` | Uses the representation of the first token (typically the [CLS] token). |
| `Max` | Takes the maximum value across all sequence positions for each dimension. |
| `Mean` | Averages all token representations across the sequence. |

