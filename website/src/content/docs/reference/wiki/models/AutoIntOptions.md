---
title: "AutoIntOptions<T>"
description: "Configuration options for AutoInt (Automatic Feature Interaction Learning)."
section: "API Reference"
---

`Options & Configuration` · `AiDotNet.Models.Options`

Configuration options for AutoInt (Automatic Feature Interaction Learning).

## For Beginners

AutoInt is designed to find interactions between features:

- **Feature embeddings**: Each feature is converted to a vector
- **Self-attention layers**: Features attend to each other to learn interactions
- **Explicit interactions**: The model learns "feature A combined with feature B"

Example use case: In click-through rate prediction, "age + product_category"
might have a strong interaction that AutoInt can automatically discover.

Example:

## How It Works

AutoInt uses multi-head self-attention to automatically learn high-order
feature interactions without manual feature engineering.

Reference: "AutoInt: Automatic Feature Interaction Learning via Self-Attentive Neural Networks" (2018)

## Properties

| Property | Summary |
|:-----|:--------|
| `AttentionDimension` | Gets or sets the attention dimension per head. |
| `CategoricalCardinalities` | Gets or sets the cardinalities of categorical features. |
| `DropoutRate` | Gets or sets the dropout rate. |
| `EmbeddingDimension` | Gets or sets the embedding dimension for features. |
| `EmbeddingInitScale` | Gets or sets the initialization scale for embeddings. |
| `HiddenActivation` | Gets or sets the hidden layer activation function. |
| `HiddenVectorActivation` | Gets or sets the hidden layer vector activation function (alternative to scalar activation). |
| `MLPHiddenDimensions` | Gets or sets the hidden dimensions for the MLP output layer. |
| `NumHeads` | Gets or sets the number of attention heads per layer. |
| `NumLayers` | Gets or sets the number of interacting (self-attention) layers. |
| `UseLayerNorm` | Gets or sets whether to use layer normalization. |
| `UseResidual` | Gets or sets whether to use residual connections. |

