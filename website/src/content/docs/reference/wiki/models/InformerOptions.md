---
title: "InformerOptions<T>"
description: "Configuration options for the Informer model (Informer: Beyond Efficient Transformer for Long Sequence Time-Series Forecasting)."
section: "API Reference"
---

`Options & Configuration` · `AiDotNet.Models.Options`

Configuration options for the Informer model (Informer: Beyond Efficient Transformer for Long Sequence Time-Series Forecasting).

## For Beginners

Informer is an efficient version of the Transformer architecture
designed specifically for long time series. Traditional transformers become very slow with long sequences,
but Informer uses smart tricks to be much faster while maintaining accuracy. It's particularly
good for forecasting that requires looking far back in history (like predicting next month based on
the past year).

## How It Works

Informer addresses the computational complexity challenges of vanilla Transformers for long-sequence forecasting.
Key innovations include:

- ProbSparse self-attention mechanism (O(L log L) complexity instead of O(L²))
- Self-attention distilling for efficient stacking
- Generative style decoder for one-forward prediction

## Properties

| Property | Summary |
|:-----|:--------|
| `BatchSize` | Gets or sets the batch size. |
| `DistillingFactor` | Gets or sets the distilling factor for self-attention distilling. |
| `DropoutRate` | Gets or sets the dropout rate. |
| `EmbeddingDim` | Gets or sets the embedding dimension. |
| `Epochs` | Gets or sets the number of training epochs. |
| `ForecastHorizon` | Gets or sets the forecast horizon (decoder output length). |
| `LearningRate` | Gets or sets the learning rate. |
| `LookbackWindow` | Gets or sets the lookback window (encoder input length). |
| `NumAttentionHeads` | Gets or sets the number of attention heads. |
| `NumDecoderLayers` | Gets or sets the number of decoder layers. |
| `NumEncoderLayers` | Gets or sets the number of encoder layers. |

