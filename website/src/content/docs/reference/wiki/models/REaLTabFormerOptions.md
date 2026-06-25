---
title: "REaLTabFormerOptions<T>"
description: "Configuration options for REaLTabFormer, a GPT-2 style autoregressive transformer for generating realistic tabular data by treating columns as a sequence of tokens."
section: "API Reference"
---

`Options & Configuration` · `AiDotNet.Models.Options`

Configuration options for REaLTabFormer, a GPT-2 style autoregressive transformer
for generating realistic tabular data by treating columns as a sequence of tokens.

## For Beginners

REaLTabFormer treats each row like a sentence and each column
value like a word. It generates data by predicting one column at a time:

1. Start with a special [START] token
2. Predict the first column's value: "Age = 35"
3. Given Age=35, predict next: "Income = 75000"
4. Given Age=35, Income=75000, predict: "Education = MS"
5. Continue until all columns are filled

This captures column dependencies naturally because each prediction
is conditioned on all previously generated columns.

Example:

## How It Works

REaLTabFormer tokenizes each column value and generates table rows left-to-right,
one column at a time, using a causal (autoregressive) transformer architecture.

Reference: "REaLTabFormer: Generating Realistic Relational and Tabular Data
using Transformers" (Solatorio and Dupriez, 2023)

## Properties

| Property | Summary |
|:-----|:--------|
| `BatchSize` | Gets or sets the training batch size. |
| `DropoutRate` | Gets or sets the dropout rate for attention and feed-forward layers. |
| `EmbeddingDimension` | Gets or sets the embedding dimension for tokens and positions. |
| `Epochs` | Gets or sets the number of training epochs. |
| `FeedForwardDimension` | Gets or sets the dimension of the feed-forward network in each transformer block. |
| `LearningRate` | Gets or sets the learning rate. |
| `NumBins` | Gets or sets the number of bins for discretizing continuous values. |
| `NumHeads` | Gets or sets the number of attention heads per layer. |
| `NumLayers` | Gets or sets the number of transformer layers. |
| `Temperature` | Gets or sets the sampling temperature for generation. |

