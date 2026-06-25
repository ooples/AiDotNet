---
title: "TabLLMGenOptions<T>"
description: "Configuration options for TabLLM-Gen, an LLM-style tabular data generator that uses schema-aware tokenization and autoregressive transformers with column prompts."
section: "API Reference"
---

`Options & Configuration` · `AiDotNet.Models.Options`

Configuration options for TabLLM-Gen, an LLM-style tabular data generator that uses
schema-aware tokenization and autoregressive transformers with column prompts.

## For Beginners

TabLLM-Gen treats table generation like text generation:

1. A row becomes a "sentence": "[Age: 35] [Income: 75000] [Education: MS]"
2. The model learns to predict each column value given previous columns
3. Column names and types serve as "instructions" to guide generation

This approach naturally captures column dependencies and produces coherent rows.

Example:

## How It Works

TabLLM-Gen adapts large language model techniques for tabular data:

- **Schema-aware tokenization**: Column names and types become special tokens
- **Column prompts**: Each column is prompted with its name/type for context
- **Autoregressive generation**: Generates one column at a time, left to right
- **Few-shot learning**: Can condition on example rows for in-context learning

Reference: "LLM-based Tabular Data Generation" (2024)

## Properties

| Property | Summary |
|:-----|:--------|
| `BatchSize` | Gets or sets the training batch size. |
| `DropoutRate` | Gets or sets the dropout rate. |
| `EmbeddingDimension` | Gets or sets the embedding dimension. |
| `Epochs` | Gets or sets the number of training epochs. |
| `FeedForwardDimension` | Gets or sets the feed-forward dimension. |
| `LearningRate` | Gets or sets the learning rate. |
| `NumBins` | Gets or sets the number of bins for discretizing continuous values. |
| `NumHeads` | Gets or sets the number of attention heads. |
| `NumLayers` | Gets or sets the number of transformer layers. |
| `SchemaTokensPerColumn` | Gets or sets the number of extra schema tokens per column. |
| `Temperature` | Gets or sets the sampling temperature for generation. |

