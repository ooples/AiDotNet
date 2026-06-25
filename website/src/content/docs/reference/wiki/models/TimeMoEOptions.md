---
title: "TimeMoEOptions<T>"
description: "Configuration options for Time-MoE (Billion-Scale Time Series Foundation Models with Mixture of Experts)."
section: "API Reference"
---

`Options & Configuration` · `AiDotNet.Models.Options`

Configuration options for Time-MoE (Billion-Scale Time Series Foundation Models with Mixture of Experts).

## For Beginners

Time-MoE achieves massive scale efficiently:

**Mixture of Experts:**
Instead of using one large feed-forward network, MoE uses multiple smaller "expert"
networks and a router that selects which experts to use for each input. This means
the model has many parameters but only uses a fraction for each prediction.

**Model Sizes:**

- 50M parameters (all active)
- 200M total / ~50M active per token
- 2.4B total / ~300M active per token (largest)

## How It Works

Time-MoE is the first billion-scale time series foundation model, using sparse Mixture
of Experts (MoE) for efficient scaling up to 2.4B parameters. It uses a decoder-only
transformer with MoE feed-forward layers.

**Reference:** Shi et al., "Time-MoE: Billion-Scale Time Series Foundation Models
with Mixture of Experts", ICLR 2025. https://openreview.net/forum?id=e1wDDFmlVu

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `TimeMoEOptions` | Initializes a new instance with default values. |
| `TimeMoEOptions(TimeMoEOptions<>)` | Initializes a new instance by copying from another instance. |

## Properties

| Property | Summary |
|:-----|:--------|
| `ContextLength` | Gets or sets the context length. |
| `DropoutRate` | Gets or sets the dropout rate. |
| `ForecastHorizon` | Gets or sets the forecast horizon. |
| `HiddenDimension` | Gets or sets the hidden dimension. |
| `IntermediateSize` | Gets or sets the intermediate size per expert. |
| `ModelSize` | Gets or sets the model size variant. |
| `NumActiveExperts` | Gets or sets the number of active experts per token. |
| `NumExperts` | Gets or sets the total number of experts in each MoE layer. |
| `NumHeads` | Gets or sets the number of attention heads. |
| `NumLayers` | Gets or sets the number of transformer layers. |
| `PatchLength` | Gets or sets the patch length. |
| `RouterAuxLossWeight` | Gets or sets the auxiliary loss weight for the router load balancing. |

