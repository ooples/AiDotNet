---
title: "LLMTimeOptions<T>"
description: "Configuration options for LLM-Time (Zero-Shot Time Series Forecasting via LLM Tokenization)."
section: "API Reference"
---

`Options & Configuration` · `AiDotNet.Models.Options`

Configuration options for LLM-Time (Zero-Shot Time Series Forecasting via LLM Tokenization).

## For Beginners

LLM-Time takes a surprising approach:

**How It Works:**

1. Convert numbers to text: [1.5, 2.3, 3.1] → "1.500, 2.300, 3.100"
2. Feed the text to a pretrained LLM (GPT-3, LLaMA)
3. The LLM predicts the next "tokens" (which are digits of future values)
4. Parse the generated text back into numbers

**Key Advantages:**

- Zero-shot: no training required at all
- Leverages the vast pattern knowledge of large language models
- Produces probabilistic forecasts via sampling

**Trade-offs:**

- Requires access to a large language model API
- Limited precision (controlled by decimal places)
- Slower than specialized time series models

## How It Works

LLM-Time converts numeric time series into text strings and uses pretrained LLMs (GPT-3, LLaMA)
for zero-shot forecasting by treating the task as next-token prediction on numerical text.
No fine-tuning is required—the LLM backbone is frozen.

**Reference:** Gruver et al., "Large Language Models Are Zero-Shot Time Series Forecasters", NeurIPS 2023.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `LLMTimeOptions` | Initializes a new instance with default values. |
| `LLMTimeOptions(LLMTimeOptions<>)` | Initializes a new instance by copying from another instance. |

## Properties

| Property | Summary |
|:-----|:--------|
| `ContextLength` | Gets or sets the maximum number of historical time steps for the text prompt. |
| `DropoutRate` | Gets or sets the dropout rate. |
| `ForecastHorizon` | Gets or sets the number of future time steps to forecast. |
| `HiddenDimension` | Gets or sets the hidden dimension of the LLM backbone. |
| `ModelSize` | Gets or sets the model size variant. |
| `NumDecimalPlaces` | Gets or sets the number of decimal places for numeric-to-text conversion. |
| `NumHeads` | Gets or sets the number of attention heads. |
| `NumLayers` | Gets or sets the number of transformer layers. |
| `NumSamples` | Gets or sets the number of samples for probabilistic forecasting. |
| `Temperature` | Gets or sets the LLM sampling temperature. |

