---
title: "ChronosBoltOptions<T>"
description: "Configuration options for Chronos-Bolt (Fast Non-Autoregressive Time Series Forecasting)."
section: "API Reference"
---

`Options & Configuration` · `AiDotNet.Models.Options`

Configuration options for Chronos-Bolt (Fast Non-Autoregressive Time Series Forecasting).

## For Beginners

Chronos-Bolt trades autoregressive generation for speed:

**Key Difference from Chronos v1/v2:**

- Chronos v1/v2: Generates one token at a time (autoregressive, slow)
- Chronos-Bolt: Generates all forecast steps at once (non-autoregressive, fast)

**Architecture:**

- Encoder: Processes the input context
- Decoder: Directly outputs all forecast quantiles in one pass
- No autoregressive loop = much faster inference

**When to Use:**

- When you need fast inference (production/real-time)
- When Chronos v1/v2 is too slow for your use case
- When you need quantile forecasts (uncertainty estimates)

## How It Works

Chronos-Bolt is part of the Amazon Chronos family but uses an encoder-decoder architecture
with direct quantile forecasting (non-autoregressive), making it significantly faster than
the autoregressive Chronos v1/v2 models while maintaining competitive accuracy.

**Reference:** Part of Chronos family, Amazon, Nov 2024.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `ChronosBoltOptions` | Initializes a new instance with default values. |
| `ChronosBoltOptions(ChronosBoltOptions<>)` | Initializes a new instance by copying from another instance. |

## Properties

| Property | Summary |
|:-----|:--------|
| `ContextLength` | Gets or sets the context length. |
| `DecoderHiddenDim` | Gets or sets the decoder hidden dimension. |
| `DropoutRate` | Gets or sets the dropout rate. |
| `EncoderHiddenDim` | Gets or sets the encoder hidden dimension. |
| `ForecastHorizon` | Gets or sets the forecast horizon. |
| `ModelSize` | Gets or sets the model size variant. |
| `NumDecoderLayers` | Gets or sets the number of decoder layers. |
| `NumEncoderLayers` | Gets or sets the number of encoder layers. |
| `NumHeads` | Gets or sets the number of attention heads. |
| `NumQuantiles` | Gets or sets the number of quantiles for direct quantile forecasting. |
| `PatchLength` | Gets or sets the patch length for input tokenization. |

