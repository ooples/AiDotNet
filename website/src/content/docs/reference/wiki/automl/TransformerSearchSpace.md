---
title: "TransformerSearchSpace<T>"
description: "Defines the Transformer-based search space for neural architecture search."
section: "API Reference"
---

`Models & Types` · `AiDotNet.AutoML.SearchSpace`

Defines the Transformer-based search space for neural architecture search.
Includes attention mechanisms, feed-forward networks, and various architectural choices.

## Properties

| Property | Summary |
|:-----|:--------|
| `AttentionHeads` | Number of attention heads to search over |
| `DropoutRates` | Dropout rates to search over |
| `FeedForwardMultipliers` | Feed-forward expansion ratios |
| `HiddenDimensions` | Hidden dimensions to consider |
| `UsePreNorm` | Whether to use Pre-LayerNorm (true) or Post-LayerNorm (false) |

