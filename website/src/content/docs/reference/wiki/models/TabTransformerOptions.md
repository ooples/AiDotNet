---
title: "TabTransformerOptions<T>"
description: "Configuration options for TabTransformer."
section: "API Reference"
---

`Options & Configuration` · `AiDotNet.Models.Options`

Configuration options for TabTransformer.

## For Beginners

TabTransformer treats categorical features specially:

Key ideas:

1. **Categorical Embeddings**: Each category value gets a learned vector
2. **Transformer for Categories**: Self-attention captures relationships between categories
3. **Numerical Features Unchanged**: Numbers pass through directly
4. **Concatenation**: Transformed categories + numbers → MLP → prediction

Why this works:

- Categorical features often have complex interactions (e.g., city + job type)
- Transformer attention can learn these interactions automatically
- Numerical features don't need the same treatment

Example:

## How It Works

TabTransformer applies transformer self-attention only to categorical features,
while keeping numerical features in their original form. The transformed categorical
embeddings are concatenated with numerical features for prediction.

Reference: "TabTransformer: Tabular Data Modeling Using Contextual Embeddings" (2020)

## Properties

| Property | Summary |
|:-----|:--------|
| `DropoutRate` | Dropout rate. |
| `EmbeddingDimension` | Gets or sets the embedding dimension for categorical features. |
| `EmbeddingInitScale` | Gets or sets the initialization scale for embeddings. |
| `FeedForwardDimension` | Gets the feed-forward network dimension. |
| `FeedForwardMultiplier` | Gets or sets the feed-forward dimension multiplier. |
| `HiddenDimension` | Hidden dimension size for transformer. |
| `MLPHiddenDimensions` | Gets or sets the hidden dimensions for the MLP head. |
| `NumCategoricalFeatures` | Gets or sets the number of categorical features. |
| `NumLayers` | Number of transformer layers. |
| `UseColumnEmbedding` | Gets or sets whether to use column embedding (add learnable column-specific vectors). |
| `UseLayerNorm` | Gets or sets whether to use layer normalization. |

## Fields

| Field | Summary |
|:-----|:--------|
| `_categoricalCardinalities` | Gets or sets the cardinalities of categorical features. |
| `_numHeads` | Number of attention heads. |

