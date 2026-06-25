---
title: "TESTOptions<T>"
description: "Configuration options for TEST (Text Embedding for Seasonality and Trend — Generating Text-Aligned Embeddings for Time Series)."
section: "API Reference"
---

`Options & Configuration` · `AiDotNet.Models.Options`

Configuration options for TEST (Text Embedding for Seasonality and Trend — Generating Text-Aligned Embeddings for Time Series).

## How It Works

TEST generates text-prototype-aligned embeddings for time series by leveraging pretrained
language model knowledge. It translates seasonal/trend patterns into text descriptions
and aligns time series embeddings to these text prototypes.

**Reference:** Sun et al., "TEST: Text Prototype Aligned Embedding to Activate LLM's Ability for Time Series", 2024.

## Properties

| Property | Summary |
|:-----|:--------|
| `AlignmentWeight` | Gets or sets the weight for text-alignment contrastive loss. |
| `NumPrototypes` | Gets or sets the number of text prototypes for alignment. |
| `TextEmbeddingDimension` | Gets or sets the text embedding dimension from the language model. |

