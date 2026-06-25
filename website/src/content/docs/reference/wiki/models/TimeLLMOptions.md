---
title: "TimeLLMOptions<T>"
description: "Configuration options for Time-LLM (Large Language Model Reprogramming for Time Series)."
section: "API Reference"
---

`Options & Configuration` · `AiDotNet.Models.Options`

Configuration options for Time-LLM (Large Language Model Reprogramming for Time Series).

## For Beginners

Time-LLM is a clever way to use powerful language models for time series:

**The Key Insight:**
LLMs like GPT/LLaMA are amazing at pattern recognition in sequences.
Time-LLM asks: "Can we make time series 'speak' the language of LLMs?"

**How It Works:**

1. **Patch Reprogramming:** Convert time series patches into "prompt-like" tokens
2. **Text Prototypes:** Learn embeddings that bridge numeric and text domains
3. **Frozen LLM:** The LLM weights stay fixed (no fine-tuning needed)
4. **Output Projection:** Map LLM output back to forecast values

**Advantages:**

- Leverages powerful pretrained LLMs without expensive fine-tuning
- Works with any LLM backbone (GPT-2, LLaMA, etc.)
- Only trains small reprogramming layers
- Zero-shot transfer to new domains

**Architecture:**
[Time Series] → [Patch] → [Reprogram] → [Frozen LLM] → [Project] → [Forecast]

## How It Works

Time-LLM repurposes frozen large language models for time series forecasting by
learning a reprogramming layer that translates time series into text-like representations
that the LLM can understand.

**Reference:** Jin et al., "Time-LLM: Time Series Forecasting by Reprogramming Large Language Models", 2024.
https://arxiv.org/abs/2310.01728

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `TimeLLMOptions` | Initializes a new instance of the `TimeLLMOptions` class with default values. |
| `TimeLLMOptions(TimeLLMOptions<>)` | Initializes a new instance by copying from another instance. |

## Properties

| Property | Summary |
|:-----|:--------|
| `ContextLength` | Gets or sets the context length (input sequence length). |
| `DropoutRate` | Gets or sets the dropout rate for regularization. |
| `ForecastHorizon` | Gets or sets the forecast horizon (prediction length). |
| `LLMBackbone` | Gets or sets the LLM backbone type. |
| `LLMDimension` | Gets or sets the LLM hidden dimension. |
| `NumHeads` | Gets or sets the number of attention heads. |
| `NumLayers` | Gets or sets the number of transformer layers in the reprogramming module. |
| `NumPrototypes` | Gets or sets the number of text prototypes. |
| `PatchLength` | Gets or sets the patch length for input segmentation. |
| `PatchStride` | Gets or sets the patch stride. |

