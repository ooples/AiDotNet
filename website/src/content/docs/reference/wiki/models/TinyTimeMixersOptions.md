---
title: "TinyTimeMixersOptions<T>"
description: "Configuration options for Tiny Time Mixers (TTM) foundation model."
section: "API Reference"
---

`Options & Configuration` · `AiDotNet.Models.Options`

Configuration options for Tiny Time Mixers (TTM) foundation model.

## For Beginners

TTM is designed to be small, fast, and surprisingly powerful:

**Key Innovation — MLP-Mixer Architecture:**
Instead of expensive attention mechanisms (like in GPT or Chronos), TTM uses simple
MLP (Multi-Layer Perceptron) blocks that mix information across two dimensions:

1. **Temporal mixing**: Exchanges information across time steps within each patch
2. **Channel mixing**: Exchanges information across different features/variables

**Why This Works:**

- MLP-Mixers are 10-100x faster than attention mechanisms
- For time series, you don't always need the flexibility of attention
- The model can be trained on much less data and compute
- Perfect for edge deployment and real-time applications

**Performance Highlights:**

- 1-5M parameters (vs 200-700M for Chronos/MOIRAI)
- Outperforms or matches much larger foundation models
- Can forecast on CPU in real-time
- Trains in minutes instead of hours/days

**Adaptive Patching:**
TTM can automatically adjust its patch size based on the input data characteristics,
allowing it to handle different frequencies without manual tuning.

## How It Works

Tiny Time Mixers (TTM) is IBM Research's compact foundation model for time series forecasting
that uses an MLP-Mixer architecture instead of attention-based transformers. Despite having
only 1-5 million parameters, TTM outperforms models 20-40x its size on standard benchmarks.

**Reference:** Ekambaram et al., "Tiny Time Mixers (TTMs): Fast Pre-trained Models
for Enhanced Zero/Few-Shot Forecasting of Multivariate Time Series", NeurIPS 2024.
https://arxiv.org/abs/2401.03955

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `TinyTimeMixersOptions` | Initializes a new instance with default values. |
| `TinyTimeMixersOptions(TinyTimeMixersOptions<>)` | Initializes a new instance by copying from another instance. |

## Properties

| Property | Summary |
|:-----|:--------|
| `ContextLength` | Gets or sets the context length (input sequence length). |
| `DropoutRate` | Gets or sets the dropout rate for regularization. |
| `ExpansionFactor` | Gets or sets the expansion factor for the mixer feed-forward networks. |
| `ForecastHorizon` | Gets or sets the forecast horizon (prediction length). |
| `HiddenDimension` | Gets or sets the hidden dimension of the mixer layers. |
| `ModelSize` | Gets or sets the model size variant. |
| `NumFeatures` | Gets or sets the number of input features (channels) for multivariate forecasting. |
| `NumMixerLayers` | Gets or sets the number of mixer layers. |
| `PatchLength` | Gets or sets the patch length for input tokenization. |
| `UseAdaptivePatching` | Gets or sets whether to use adaptive patching. |

