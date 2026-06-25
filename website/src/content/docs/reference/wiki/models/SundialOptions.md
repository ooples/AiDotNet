---
title: "SundialOptions<T>"
description: "Configuration options for Sundial (A Family of Highly Capable Time Series Foundation Models)."
section: "API Reference"
---

`Options & Configuration` · `AiDotNet.Models.Options`

Configuration options for Sundial (A Family of Highly Capable Time Series Foundation Models).

## For Beginners

Sundial is a highly efficient forecasting model:

**Key Innovation:**
Sundial outperforms Time-MoE (which has up to 2.4B params) with significantly fewer
parameters, achieving a 4.71% average MSE reduction.

**Architecture:**

- Decoder-only transformer (like GPT)
- Patch-based input tokenization
- Autoregressive generation for forecasting
- Efficient scaling through architectural improvements

**When to Use:**

- High-accuracy forecasting with moderate compute
- When you need better accuracy than Time-MoE with fewer parameters
- General-purpose time series forecasting across domains

## How It Works

Sundial is a decoder-only time series foundation model that achieves state-of-the-art
performance with fewer parameters than competing models like Time-MoE. It uses a
GPT-style autoregressive architecture with patch-based tokenization.

**Reference:** "Sundial: A Family of Highly Capable Time Series Foundation Models", 2025.
https://arxiv.org/abs/2502.00816

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `SundialOptions` | Initializes a new instance with default values. |
| `SundialOptions(SundialOptions<>)` | Initializes a new instance by copying from another instance. |

## Properties

| Property | Summary |
|:-----|:--------|
| `ContextLength` | Gets or sets the context length (input sequence length). |
| `DropoutRate` | Gets or sets the dropout rate for regularization. |
| `ForecastHorizon` | Gets or sets the forecast horizon (prediction length). |
| `HiddenDimension` | Gets or sets the hidden dimension of the transformer. |
| `IntermediateSize` | Gets or sets the intermediate size for the feed-forward network. |
| `ModelSize` | Gets or sets the model size variant. |
| `NumHeads` | Gets or sets the number of attention heads. |
| `NumLayers` | Gets or sets the number of transformer layers. |
| `NumQuantiles` | Gets or sets the number of quantiles for probabilistic forecasting. |
| `PatchLength` | Gets or sets the patch length for input tokenization. |
| `UseFlashAttention` | Gets or sets whether to use flash attention for efficient computation. |

