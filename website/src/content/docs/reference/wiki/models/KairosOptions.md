---
title: "KairosOptions<T>"
description: "Configuration options for Kairos (Adaptive and Generalizable Time Series Foundation Model)."
section: "API Reference"
---

`Options & Configuration` · `AiDotNet.Models.Options`

Configuration options for Kairos (Adaptive and Generalizable Time Series Foundation Model).

## For Beginners

Kairos adapts its tokenization to your data:

**Adaptive Tokenization:**
Unlike fixed-size patching, Kairos uses multiple patch sizes simultaneously and
a learned router decides which granularity is best for each segment. Dense/volatile
regions get fine-grained tokens; smooth regions get coarse tokens.

**Mixture-of-Size Encoder:**
Multiple encoder branches process patches at different sizes, then a gating
mechanism combines the results based on local information density.

## How It Works

Kairos uses a Mixture-of-Size Encoder with adaptive tokenization that adjusts patch
granularity based on local information density. This parameter-efficient approach
handles diverse time series characteristics without fixed tokenization.

**Reference:** "Kairos: Towards Adaptive and Generalizable Time Series Foundation Models", 2025.
https://arxiv.org/abs/2509.25826

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `KairosOptions` | Initializes a new instance with default values. |
| `KairosOptions(KairosOptions<>)` | Initializes a new instance by copying from another instance. |

## Properties

| Property | Summary |
|:-----|:--------|
| `ContextLength` | Gets or sets the context length. |
| `DropoutRate` | Gets or sets the dropout rate. |
| `ForecastHorizon` | Gets or sets the forecast horizon. |
| `HiddenDimension` | Gets or sets the hidden dimension. |
| `IntermediateSize` | Gets or sets the intermediate size for the feed-forward network. |
| `ModelSize` | Gets or sets the model size variant. |
| `NumHeads` | Gets or sets the number of attention heads. |
| `NumLayers` | Gets or sets the number of transformer layers. |
| `PatchSizes` | Gets or sets the multiple patch sizes for adaptive tokenization. |

