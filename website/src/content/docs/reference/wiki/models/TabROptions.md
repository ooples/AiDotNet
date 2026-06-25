---
title: "TabROptions<T>"
description: "Configuration options for TabR, a retrieval-augmented model for tabular data."
section: "API Reference"
---

`Options & Configuration` · `AiDotNet.Models.Options`

Configuration options for TabR, a retrieval-augmented model for tabular data.

## For Beginners

TabR is like a student who looks at similar past problems
to help solve a new one.

How it works:

1. **Encode**: Convert features to a learned representation
2. **Retrieve**: Find the K most similar training samples
3. **Attend**: Use attention to aggregate information from neighbors
4. **Predict**: Make a prediction using both query and neighbor information

Why this works well for tabular data:

- Tabular data often has local structure (similar inputs → similar outputs)
- Retrieval adds "memory" without needing to memorize in network weights
- Combines strengths of neural networks (feature learning) and k-NN (locality)
- Naturally handles rare patterns by finding similar examples

Example usage:

## How It Works

TabR combines neural networks with instance-based learning by retrieving similar
training examples and using their information to help make predictions. It encodes
both the query sample and retrieved neighbors, then aggregates the information
using attention.

Reference: "TabR: Tabular Deep Learning Meets Nearest Neighbors" (2023)

## Properties

| Property | Summary |
|:-----|:--------|
| `ActivationType` | Gets or sets the activation function type. |
| `DropoutRate` | Gets or sets the dropout rate. |
| `EmbeddingDimension` | Gets or sets the embedding dimension for encoding features. |
| `EnableGradientClipping` | Gets or sets whether to enable gradient clipping. |
| `FeedForwardDimension` | Gets the feed-forward network dimension. |
| `FeedForwardMultiplier` | Gets or sets the feed-forward dimension multiplier. |
| `IncludeNeighborTargets` | Gets or sets whether to include target values of retrieved neighbors. |
| `MaxGradientNorm` | Gets or sets the maximum gradient norm for clipping. |
| `NormalizeEmbeddings` | Gets or sets whether to normalize embeddings for retrieval. |
| `NumAttentionHeads` | Gets or sets the number of attention heads for neighbor aggregation. |
| `NumContextLayers` | Gets or sets the number of context encoder layers. |
| `NumLayers` | Gets or sets the number of MLP layers in the feature encoder. |
| `NumNeighbors` | Gets or sets the number of nearest neighbors to retrieve. |
| `RetrievalTemperature` | Gets or sets the temperature for retrieval softmax. |
| `UseFiLM` | Gets or sets whether to use feature-wise linear modulation. |
| `UseLayerNorm` | Gets or sets whether to use layer normalization. |
| `WeightDecay` | Gets or sets the weight decay coefficient. |

## Methods

| Method | Summary |
|:-----|:--------|
| `Clone` | Creates a copy of the options. |

