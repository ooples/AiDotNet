---
title: "ChronosFinanceOptions<T>"
description: "Configuration options for Chronos Finance (Amazon's time series foundation model for financial forecasting)."
section: "API Reference"
---

`Options & Configuration` · `AiDotNet.Models.Options`

Configuration options for Chronos Finance (Amazon's time series foundation model for financial forecasting).

## For Beginners

Chronos treats time series forecasting as a language modeling problem:

**The Tokenization Idea:**
Just like text is converted to tokens for GPT, Chronos converts time series:

- Scales values to a standard range (e.g., [-1, 1])
- Quantizes into discrete bins (e.g., 4096 bins)
- Each bin becomes a "token" like words in text

**Why Tokenize?**

- Leverages powerful pretrained language models (T5, GPT)
- No need to design specialized time series architectures
- Benefits from LLM's pattern recognition capabilities
- Easy to handle different scales and magnitudes

**Probabilistic via Sampling:**
Like GPT generating text, Chronos samples from the predicted token distribution:

- Each token prediction is a probability over all bins
- Multiple samples give uncertainty estimates
- More diverse samples = higher uncertainty

**Architecture Variants:**
Chronos comes in different sizes (like GPT-2 vs GPT-3):

- Mini: Fast, lightweight
- Small: Balanced
- Base: More capacity
- Large: Best accuracy, more compute

## How It Works

Chronos Finance is an implementation of Amazon's Chronos foundation model optimized for
financial time series forecasting. It tokenizes time series values using scaling and quantization,
then uses a language model to generate probabilistic forecasts.

**Reference:** Ansari et al., "Chronos: Learning the Language of Time Series", 2024.
https://arxiv.org/abs/2403.07815

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `ChronosFinanceOptions` | Initializes a new instance of the `ChronosFinanceOptions` class with default values. |
| `ChronosFinanceOptions(ChronosFinanceOptions<>)` | Initializes a new instance by copying from another instance. |

## Properties

| Property | Summary |
|:-----|:--------|
| `ContextLength` | Gets or sets the context length (input sequence length). |
| `DropoutRate` | Gets or sets the dropout rate for regularization. |
| `ForecastHorizon` | Gets or sets the forecast horizon (prediction length). |
| `GroupAttentionGroups` | Gets or sets the number of groups for group attention in Chronos-2. |
| `HiddenDimension` | Gets or sets the hidden dimension of the transformer. |
| `IntermediateSize` | Gets or sets the intermediate size for the feed-forward network. |
| `MaxContextLength` | Gets or sets the maximum context length for Chronos-2 encoder-only mode. |
| `ModelSize` | Gets or sets the model size variant. |
| `NumCovariates` | Gets or sets the number of exogenous covariates for Chronos-2 multivariate support. |
| `NumHeads` | Gets or sets the number of attention heads. |
| `NumLayers` | Gets or sets the number of transformer layers. |
| `NumQuantiles` | Gets or sets the number of quantiles for Chronos-2 quantile forecasting. |
| `NumSamples` | Gets or sets the number of forecast samples for uncertainty estimation. |
| `NumTokens` | Gets or sets the number of discrete tokens (bins) for quantization. |
| `Temperature` | Gets or sets the temperature for sampling. |
| `UseMultivariate` | Gets or sets whether to enable multivariate forecasting in Chronos-2. |
| `UsePatchInput` | Gets or sets whether to use patch-based input for Chronos-2. |

