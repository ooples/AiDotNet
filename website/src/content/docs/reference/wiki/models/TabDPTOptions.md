---
title: "TabDPTOptions<T>"
description: "Configuration options for TabDPT (Tabular Data Pre-Training)."
section: "API Reference"
---

`Options & Configuration` · `AiDotNet.Models.Options`

Configuration options for TabDPT (Tabular Data Pre-Training).

## For Beginners

TabDPT brings the power of foundation models to tabular data:

- **Pre-training**: The model learns patterns from many tabular datasets
- **Transfer learning**: These patterns help on new, unseen datasets
- **In-context learning**: Can adapt to new tasks without fine-tuning

Example:

## How It Works

TabDPT is a foundation model approach for tabular data that uses pre-training
on diverse datasets to learn transferable representations.

Reference: "TabDPT: Scaling Tabular Foundation Models" (2025)

## Properties

| Property | Summary |
|:-----|:--------|
| `CategoricalCardinalities` | Gets or sets the cardinalities of categorical features. |
| `ContextLength` | Gets or sets the context length (number of examples for in-context learning). |
| `DropoutRate` | Gets or sets the dropout rate. |
| `EmbeddingDimension` | Gets or sets the embedding dimension. |
| `FeedForwardDimension` | Gets the feed-forward dimension. |
| `FeedForwardMultiplier` | Gets or sets the feed-forward dimension multiplier. |
| `HiddenActivation` | Gets or sets the hidden layer activation function. |
| `HiddenVectorActivation` | Gets or sets the hidden layer vector activation function (alternative to scalar activation). |
| `InitScale` | Gets or sets the initialization scale. |
| `InputActivation` | Gets or sets the input projection activation function. |
| `MaxFeatures` | Gets or sets the maximum number of features supported. |
| `NumHeads` | Gets or sets the number of attention heads. |
| `NumLayers` | Gets or sets the number of transformer layers. |
| `OutputHeadDimensions` | Gets or sets the hidden dimensions for the output head. |
| `UseFeatureAttention` | Gets or sets whether to use feature-wise attention. |
| `UseLayerNorm` | Gets or sets whether to use layer normalization. |
| `UsePreNorm` | Gets or sets whether to use pre-norm (norm before attention). |

