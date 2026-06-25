---
title: "TimerOptions<T>"
description: "Configuration options for Timer (Generative Pre-Training for Time Series)."
section: "API Reference"
---

`Options & Configuration` · `AiDotNet.Models.Options`

Configuration options for Timer (Generative Pre-Training for Time Series).

## For Beginners

Timer brings GPT-style pre-training to time series:

**The Key Insight:**
Just like GPT learns language by predicting the next token, Timer learns
time series patterns by predicting future values. Pre-training on diverse
datasets enables strong zero-shot transfer.

**How It Works:**

1. **Autoregressive Pre-training:** Learn to predict future from past
2. **Masked Modeling:** Learn to reconstruct masked portions
3. **Multi-scale Processing:** Handle different temporal granularities
4. **Fine-tuning:** Adapt to specific domains with minimal data

**Architecture:**

- Patch-based tokenization (like PatchTST)
- GPT-style decoder transformer
- Autoregressive generation head
- Optional masked modeling objective

**Advantages:**

- Strong zero-shot and few-shot performance
- Generalizes across domains and frequencies
- Efficient fine-tuning with minimal labeled data
- Handles variable sequence lengths

## How It Works

Timer is a generative pre-training approach for time series that uses
autoregressive generation combined with masked modeling to learn rich
temporal representations from diverse time series datasets.

**Reference:** Liu et al., "Timer: Generative Pre-Training of Time Series", 2024.
https://arxiv.org/abs/2402.02368

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `TimerOptions` | Initializes a new instance of the `TimerOptions` class with default values. |
| `TimerOptions(TimerOptions<>)` | Initializes a new instance by copying from another instance. |

## Properties

| Property | Summary |
|:-----|:--------|
| `ContextLength` | Gets or sets the context length (input sequence length). |
| `DropoutRate` | Gets or sets the dropout rate for regularization. |
| `ForecastHorizon` | Gets or sets the forecast horizon (prediction length). |
| `GenerationTemperature` | Gets or sets the temperature for sampling during generation. |
| `HiddenDimension` | Gets or sets the hidden dimension size. |
| `MaskRatio` | Gets or sets the mask ratio for masked modeling pre-training. |
| `MaxContextLength` | Gets or sets the maximum context length for Timer-XL. |
| `ModelSize` | Gets or sets the model size variant for Timer-XL. |
| `NumHeads` | Gets or sets the number of attention heads. |
| `NumLayers` | Gets or sets the number of transformer layers. |
| `PatchLength` | Gets or sets the patch length for tokenization. |
| `PatchStride` | Gets or sets the patch stride. |
| `UseAutoregressiveDecoding` | Gets or sets whether to use autoregressive decoding during generation. |

