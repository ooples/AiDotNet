---
title: "SAINTOptions<T>"
description: "Configuration options for SAINT (Self-Attention and Intersample Attention Transformer)."
section: "API Reference"
---

`Options & Configuration` · `AiDotNet.Models.Options`

Configuration options for SAINT (Self-Attention and Intersample Attention Transformer).

## For Beginners

SAINT is powerful because it learns two things:

- **Column attention**: Which features are related to each other?
- **Row attention**: Which training samples are similar?

This dual attention makes SAINT especially good when similar samples
share patterns that are useful for prediction.

Example:

## How It Works

SAINT combines two types of attention:

1. Self-attention over features (column attention, like FT-Transformer)
2. Inter-sample attention (row attention, comparing samples within a batch)

Reference: "SAINT: Improved Neural Networks for Tabular Data via Row Attention
and Contrastive Pre-Training" (2021)

## Properties

| Property | Summary |
|:-----|:--------|
| `AttentionDropoutRate` | Gets or sets the attention dropout rate (separate from general dropout). |
| `BatchSize` | Batch size (sequence length for inter-sample attention). |
| `CategoricalCardinalities` | Gets or sets the cardinalities of categorical features. |
| `DropoutRate` | Dropout rate. |
| `EmbeddingDimension` | Gets or sets the embedding dimension for features. |
| `EmbeddingInitScale` | Gets or sets the initialization scale for embeddings. |
| `FeedForwardDimension` | Gets the feed-forward network dimension. |
| `FeedForwardMultiplier` | Gets or sets the feed-forward dimension multiplier. |
| `HiddenActivation` | Gets or sets the hidden layer activation function. |
| `HiddenDimension` | Hidden dimension size for feed-forward networks. |
| `HiddenVectorActivation` | Gets or sets the hidden layer vector activation function (alternative to scalar activation). |
| `MLPHiddenDimensions` | Gets or sets the hidden dimensions for the MLP head. |
| `NumHeads` | Number of attention heads. |
| `NumLayers` | Number of transformer layers. |
| `UseIntersampleAttention` | Gets or sets whether to use inter-sample (row) attention. |
| `UseLayerNorm` | Gets or sets whether to use layer normalization. |
| `UsePreNorm` | Gets or sets whether to use pre-norm (norm before attention) or post-norm. |

## Methods

| Method | Summary |
|:-----|:--------|
| `Validate` | Validates that the options are consistent (e.g., NumHeads evenly divides EmbeddingDimension). |

