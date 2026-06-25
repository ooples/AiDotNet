---
title: "TETNPOptions<T, TInput, TOutput>"
description: "Configuration options for Translation-Equivariant Transformer Neural Process (TE-TNP)."
section: "API Reference"
---

`Options & Configuration` · `AiDotNet.MetaLearning.Options`

Configuration options for Translation-Equivariant Transformer Neural Process (TE-TNP).
Extends TNP with relative positional encodings for translation equivariance.

## For Beginners

TE-TNP predicts functions like TNP but ensures that shifting
all inputs by the same amount shifts all outputs correspondingly. This is achieved by
using relative distances between points rather than absolute positions in the attention mechanism.

## Properties

| Property | Summary |
|:-----|:--------|
| `EquivarianceRegWeight` | Weight for the equivariance regularization loss that encourages translation-equivariant behavior during training. |
| `NumFrequencyBands` | Number of frequency bands for sinusoidal relative positional encoding. |
| `NumHeads` | Number of attention heads for the equivariant self-attention. |
| `RepresentationDim` | Gets or sets the representation dimensionality. |

