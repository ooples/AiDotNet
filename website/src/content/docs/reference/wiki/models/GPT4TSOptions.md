---
title: "GPT4TSOptions<T>"
description: "Configuration options for GPT4TS (One Fits All: Power General Time Series Analysis by Pretrained LM)."
section: "API Reference"
---

`Options & Configuration` · `AiDotNet.Models.Options`

Configuration options for GPT4TS (One Fits All: Power General Time Series Analysis by Pretrained LM).

## For Beginners

GPT4TS repurposes GPT-2 (a language model) for time series:

**How It Works:**

1. Time series are split into patches (like "words")
2. Patches are fed through a frozen GPT-2 backbone (no weight updates)
3. A lightweight task-specific head is trained on top

**Key Advantages:**

- Leverages pretrained language model knowledge
- Only the small task head needs training (fast + low data)
- Supports multiple tasks: forecasting, classification, anomaly detection

**When to Use:**

- When you have limited training data (the frozen backbone provides strong priors)
- When you need multi-task support from a single model

## How It Works

GPT4TS uses a frozen GPT-2 backbone with task-specific heads for time series forecasting,
classification, and anomaly detection. It demonstrates that pretrained language models
transfer effectively to time series tasks without fine-tuning the backbone.

**Reference:** Zhou et al., "One Fits All: Power General Time Series Analysis by Pretrained LM", 2023.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `GPT4TSOptions` | Initializes a new instance with default values. |
| `GPT4TSOptions(GPT4TSOptions<>)` | Initializes a new instance by copying from another instance. |

## Properties

| Property | Summary |
|:-----|:--------|
| `ContextLength` | Gets or sets the number of historical time steps used as input context. |
| `DropoutRate` | Gets or sets the dropout rate for regularization. |
| `ForecastHorizon` | Gets or sets the number of future time steps to forecast. |
| `FreezeBackbone` | Gets or sets whether to freeze the GPT-2 backbone weights. |
| `HiddenDimension` | Gets or sets the hidden dimension of the GPT-2 backbone. |
| `ModelSize` | Gets or sets the model size variant. |
| `NumHeads` | Gets or sets the number of attention heads. |
| `NumLayers` | Gets or sets the number of transformer layers. |
| `PatchLength` | Gets or sets the patch length for input tokenization. |
| `Task` | Gets or sets the downstream task for the model. |

