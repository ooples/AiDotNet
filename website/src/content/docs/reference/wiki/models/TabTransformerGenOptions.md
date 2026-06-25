---
title: "TabTransformerGenOptions<T>"
description: "Configuration options for TabTransformer-Gen, a generative model that uses contextual embeddings from multi-head self-attention over categorical columns to generate realistic tabular data."
section: "API Reference"
---

`Options & Configuration` · `AiDotNet.Models.Options`

Configuration options for TabTransformer-Gen, a generative model that uses contextual embeddings
from multi-head self-attention over categorical columns to generate realistic tabular data.

## For Beginners

TabTransformer-Gen works like a fill-in-the-blank game:

1. Each categorical column gets its own "word" (embedding)
2. These words "talk to each other" through attention (e.g., "Occupation"

pays attention to "Education" because they're related)

3. During training, we randomly mask some columns and ask the model to guess them
4. During generation, we start with everything masked and iteratively fill in columns

Example:

## How It Works

TabTransformer-Gen adapts the TabTransformer architecture for data generation:

- Categorical columns get learned embeddings that attend to each other
- Continuous columns pass through with optional normalization
- A masked prediction objective trains the model to reconstruct missing columns
- Generation uses iterative masked prediction (like masked language modeling)

Reference: "TabTransformer: Tabular Data Modeling Using Contextual Embeddings"
(Huang et al., 2020) — adapted for generation with masked prediction

## Properties

| Property | Summary |
|:-----|:--------|
| `BatchSize` | Gets or sets the training batch size. |
| `DropoutRate` | Gets or sets the dropout rate. |
| `EmbeddingDimension` | Gets or sets the embedding dimension for each column. |
| `Epochs` | Gets or sets the number of training epochs. |
| `FeedForwardDimension` | Gets or sets the feed-forward dimension in each transformer block. |
| `GenerationSteps` | Gets or sets the number of iterative refinement steps during generation. |
| `LearningRate` | Gets or sets the learning rate. |
| `MaskRatio` | Gets or sets the fraction of columns to mask during training. |
| `NumHeads` | Gets or sets the number of attention heads per layer. |
| `NumLayers` | Gets or sets the number of transformer layers. |
| `VGMModes` | Gets or sets the number of VGM modes for continuous column transformation. |

