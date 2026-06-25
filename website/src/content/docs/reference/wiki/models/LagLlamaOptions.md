---
title: "LagLlamaOptions<T>"
description: "Configuration options for Lag-Llama (Large Language Model for time series forecasting)."
section: "API Reference"
---

`Options & Configuration` · `AiDotNet.Models.Options`

Configuration options for Lag-Llama (Large Language Model for time series forecasting).

## For Beginners

Lag-Llama brings large language model innovations to time series:

**The Lag Feature Idea:**
Instead of just using recent values, Lag-Llama looks at values from specific past points:

- Lag-1: Value from 1 step ago (yesterday)
- Lag-7: Value from 7 steps ago (last week, same day)
- Lag-365: Value from 365 steps ago (last year, same day)

These "lag features" help capture patterns at different time scales.

**Why Llama-style Architecture?**
Llama introduced efficient transformer improvements:

- RMSNorm: Simpler, faster layer normalization
- RoPE: Rotary Position Embeddings for better position encoding
- SwiGLU: Improved activation function
- Grouped Query Attention: More efficient attention

**Zero-Shot Capability:**
Like other foundation models, Lag-Llama is pre-trained on diverse time series
and can forecast new series without fine-tuning.

**Probabilistic Forecasting:**
Lag-Llama outputs a distribution (not just point estimates):

- Predicts parameters of a probability distribution
- Allows uncertainty quantification
- Enables risk-aware decision making

**When to Use:**

- Time series with multiple seasonal patterns (daily, weekly, yearly)
- When you need uncertainty estimates
- Cross-domain zero-shot forecasting

## How It Works

Lag-Llama is a foundation model that adapts LLM architecture for time series forecasting.
It uses a decoder-only transformer with lag-based features to capture temporal dependencies
across multiple time scales.

**Reference:** Rasul et al., "Lag-Llama: Towards Foundation Models for Probabilistic Time Series Forecasting", 2024.
https://arxiv.org/abs/2310.08278

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `LagLlamaOptions` | Initializes a new instance of the `LagLlamaOptions` class with default values. |
| `LagLlamaOptions(LagLlamaOptions<>)` | Initializes a new instance by copying from another instance. |

## Properties

| Property | Summary |
|:-----|:--------|
| `ContextLength` | Gets or sets the context length (input sequence length). |
| `DistributionOutput` | Gets or sets the distribution type for probabilistic output. |
| `DropoutRate` | Gets or sets the dropout rate for regularization. |
| `ForecastHorizon` | Gets or sets the forecast horizon (prediction length). |
| `HiddenDimension` | Gets or sets the hidden dimension of the transformer. |
| `IntermediateSize` | Gets or sets the intermediate size for the feed-forward network. |
| `LagIndices` | Gets or sets the lag indices used for feature extraction. |
| `NumHeads` | Gets or sets the number of attention heads. |
| `NumLayers` | Gets or sets the number of transformer layers. |
| `UseRoPE` | Gets or sets whether to use Rotary Position Embeddings (RoPE). |

