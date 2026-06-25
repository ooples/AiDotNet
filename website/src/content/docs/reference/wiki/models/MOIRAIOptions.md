---
title: "MOIRAIOptions<T>"
description: "Configuration options for MOIRAI (Salesforce's Universal Time Series Foundation Model)."
section: "API Reference"
---

`Options & Configuration` · `AiDotNet.Models.Options`

Configuration options for MOIRAI (Salesforce's Universal Time Series Foundation Model).

## For Beginners

MOIRAI is designed to be a truly universal time series model:

**Key Innovations:**

- Multi-patch embeddings: Uses multiple patch sizes simultaneously
- Any-variate forecasting: Handles univariate and multivariate series
- Distribution mixture outputs: Combines multiple distributions for flexibility
- Unified masked encoder: Single architecture for all forecasting tasks

**Architecture:**

1. Multi-scale patching: Creates tokens at different time scales
2. Masked encoder: Transformer encoder with masking
3. Distribution head: Outputs mixture of distributions
4. Any-to-any: Can predict any horizon from any context

**Model Sizes:**

- Small: Lightweight (parameters: ~14M)
- Base: Balanced (parameters: ~91M)
- Large: Maximum capacity (parameters: ~311M)

## How It Works

MOIRAI (Masked EncOder-based UnIveRsAl TIme Series Foundation Model) is Salesforce's
foundation model for universal time series forecasting. It uses masked encoder-based
training with multiple patches to handle any-to-any forecasting across different
frequencies and domains without fine-tuning.

**Reference:** Woo et al., "Unified Training of Universal Time Series Forecasting Transformers", 2024.
https://arxiv.org/abs/2402.02592

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `MOIRAIOptions` | Initializes a new instance of the `MOIRAIOptions` class with default values. |
| `MOIRAIOptions(MOIRAIOptions<>)` | Initializes a new instance by copying from another instance. |

## Properties

| Property | Summary |
|:-----|:--------|
| `ContextLength` | Gets or sets the context length (input sequence length). |
| `DropoutRate` | Gets or sets the dropout rate for regularization. |
| `ForecastHorizon` | Gets or sets the forecast horizon (prediction length). |
| `HiddenDimension` | Gets or sets the hidden dimension of the transformer. |
| `IntermediateSize` | Gets or sets the intermediate size for the feed-forward network. |
| `MaskRatio` | Gets or sets the mask ratio for training. |
| `ModelSize` | Gets or sets the model size variant. |
| `MultiTokenSteps` | Gets or sets the number of multi-token prediction steps for Moirai 2.0. |
| `NumHeads` | Gets or sets the number of attention heads. |
| `NumLayers` | Gets or sets the number of transformer layers. |
| `NumMixtures` | Gets or sets the number of mixture components for distribution output. |
| `NumQuantiles` | Gets or sets the number of quantiles for Moirai 2.0 quantile forecasting. |
| `PatchSize` | Gets or sets the unified patch size for Moirai 2.0. |
| `PatchSizes` | Gets or sets the patch sizes for multi-scale patching. |
| `UseDecoderOnly` | Gets or sets whether to use the decoder-only architecture (Moirai 2.0 default). |

