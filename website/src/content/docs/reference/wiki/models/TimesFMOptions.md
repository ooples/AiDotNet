---
title: "TimesFMOptions<T>"
description: "Configuration options for the TimesFM (Time Series Foundation Model)."
section: "API Reference"
---

`Options & Configuration` · `AiDotNet.Models.Options`

Configuration options for the TimesFM (Time Series Foundation Model).

## For Beginners

TimesFM is a pre-trained model for general-purpose forecasting:

**What is a Foundation Model?**
Like GPT for text, foundation models for time series are:

- Pre-trained on massive, diverse datasets
- Can generalize to new forecasting tasks without fine-tuning
- Work across different domains (finance, weather, retail, etc.)

**Zero-Shot Forecasting:**
TimesFM can forecast a time series it has never seen before:

- No training required on your specific data
- Just provide historical values and get predictions
- Works because it learned general patterns during pre-training

**Architecture:**
TimesFM uses a decoder-only transformer (like GPT):

- Processes historical time steps as "tokens"
- Each token attends to all previous tokens
- Generates forecast tokens autoregressively

**Input Patching:**
Instead of processing one time step at a time, TimesFM:

- Groups consecutive time steps into "patches"
- Each patch becomes one input token
- Reduces sequence length, enables longer context

**When to Use:**

- Quick forecasting without model training
- New domains with limited historical data
- Baseline comparisons for custom models

## How It Works

TimesFM is Google's foundation model for time series forecasting. It uses a decoder-only
transformer architecture pre-trained on a massive dataset of diverse time series, enabling
zero-shot and few-shot forecasting across different domains without fine-tuning.

**Reference:** Das et al., "A decoder-only foundation model for time-series forecasting", 2024.
https://arxiv.org/abs/2310.10688

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `TimesFMOptions` | Initializes a new instance of the `TimesFMOptions` class with default values. |
| `TimesFMOptions(TimesFMOptions<>)` | Initializes a new instance by copying from another instance. |

## Properties

| Property | Summary |
|:-----|:--------|
| `ContextLength` | Gets or sets the context length (input sequence length). |
| `DropoutRate` | Gets or sets the dropout rate for regularization. |
| `ForecastHorizon` | Gets or sets the forecast horizon (prediction length). |
| `HiddenDimension` | Gets or sets the hidden dimension of the transformer. |
| `MaxContextLength` | Gets or sets the maximum context length for TimesFM 2.5. |
| `NumHeads` | Gets or sets the number of attention heads. |
| `NumLayers` | Gets or sets the number of transformer layers. |
| `NumQuantiles` | Gets or sets the number of quantiles for TimesFM 2.5 continuous quantile forecasting. |
| `OutputPatchLength` | Gets or sets the output patch length for the per-patch forecast head. |
| `PatchLength` | Gets or sets the patch length for input tokenization. |
| `QuantileHeadDimension` | Gets or sets the hidden dimension for the quantile forecasting head. |
| `UsePretrainedWeights` | Gets or sets whether to use pre-trained weights. |

