---
title: "FTTransformerOptions<T>"
description: "Configuration options for FT-Transformer, a Feature Tokenizer + Transformer for tabular data."
section: "API Reference"
---

`Options & Configuration` · `AiDotNet.Models.Options`

Configuration options for FT-Transformer, a Feature Tokenizer + Transformer for tabular data.

## For Beginners

FT-Transformer is a way to use the powerful Transformer architecture
(the technology behind ChatGPT) on traditional tabular data like spreadsheets.

Key concepts:

- **Feature Tokenization**: Each column in your data becomes a "token" (like words in text)
- **[CLS] Token**: A special token added to aggregate information for the final prediction
- **Self-Attention**: Allows the model to learn relationships between different features
- **No Manual Feature Engineering**: The model learns which features interact automatically

Advantages:

- Captures complex feature interactions through attention
- Works well with both numerical and categorical features
- Often outperforms gradient boosting on larger datasets
- Provides attention weights for interpretability

Example usage:

## How It Works

FT-Transformer applies the transformer architecture to tabular data by treating each feature
as a token. It tokenizes numerical and categorical features into embeddings and processes
them with standard transformer encoder layers.

Reference: "Revisiting Deep Learning Models for Tabular Data" (Gorishniy et al., NeurIPS 2021)

## Properties

| Property | Summary |
|:-----|:--------|
| `AttentionDropoutRate` | Gets or sets the dropout rate specifically for attention weights. |
| `CategoricalCardinalities` | Gets or sets the categorical feature cardinalities (number of unique values per categorical feature). |
| `DropoutRate` | Gets or sets the dropout rate for attention and feed-forward layers. |
| `EmbeddingDimension` | Gets or sets the embedding dimension for feature tokens. |
| `EmbeddingInitScale` | Gets or sets the initialization scale for embeddings. |
| `EnableGradientClipping` | Gets or sets whether to enable gradient clipping. |
| `FeedForwardDimension` | Gets the feed-forward network dimension. |
| `FeedForwardMultiplier` | Gets or sets the feed-forward network dimension multiplier. |
| `LayerNormEpsilon` | Gets or sets the epsilon value for layer normalization numerical stability. |
| `MaxGradientNorm` | Gets or sets the maximum gradient norm for clipping. |
| `NumHeads` | Gets or sets the number of attention heads. |
| `NumLayers` | Gets or sets the number of transformer encoder layers. |
| `ResidualDropoutRate` | Gets or sets the dropout rate applied to the residual connections. |
| `UseNumericalBias` | Gets or sets whether to use a bias term in the numerical feature embedding. |
| `UsePreLayerNorm` | Gets or sets whether to use layer normalization before attention (Pre-LN) or after (Post-LN). |
| `UseReGLU` | Gets or sets whether to use ReGLU activation in the feed-forward network. |
| `WeightDecay` | Gets or sets the weight decay (L2 regularization) coefficient. |

## Methods

| Method | Summary |
|:-----|:--------|
| `Clone` | Creates a copy of the options. |

