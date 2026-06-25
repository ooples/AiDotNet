---
title: "MGTSDOptions<T>"
description: "Configuration options for MG-TSD (Multi-Granularity Time Series Diffusion)."
section: "API Reference"
---

`Options & Configuration` · `AiDotNet.Models.Options`

Configuration options for MG-TSD (Multi-Granularity Time Series Diffusion).

## For Beginners

MG-TSD improves on standard diffusion models by:

**Multi-Granularity Guidance:**
Instead of denoising at a single resolution, MG-TSD processes the time series
at multiple temporal granularities (e.g., hourly, daily, weekly). Coarser
predictions capture long-range trends and guide finer predictions, resulting
in more coherent forecasts across time scales.

**Key Advantages:**

- Better captures patterns at different time scales
- Coarse-to-fine guidance improves forecast coherence
- Produces calibrated probabilistic forecasts

## How It Works

MG-TSD introduces a multi-granularity guidance diffusion model that captures temporal
patterns at different scales. It uses a coarse-to-fine guidance mechanism where
predictions at coarser granularities guide the fine-grained diffusion process.

**Reference:** Fan et al., "MG-TSD: Multi-Granularity Time Series Diffusion Models with Guided Learning Process", ICLR 2024.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `MGTSDOptions` | Initializes a new instance with default values. |
| `MGTSDOptions(MGTSDOptions<>)` | Initializes a new instance by copying from another instance. |

## Properties

| Property | Summary |
|:-----|:--------|
| `BetaEnd` | Gets or sets the ending beta value for the noise schedule. |
| `BetaStart` | Gets or sets the starting beta value for the noise schedule. |
| `ContextLength` | Gets or sets the number of historical time steps used as input context. |
| `DiffusionSteps` | Gets or sets the number of diffusion (denoising) steps. |
| `DropoutRate` | Gets or sets the dropout rate for regularization. |
| `ForecastHorizon` | Gets or sets the number of future time steps to forecast. |
| `GuidanceWeight` | Gets or sets the weight for cross-granularity guidance. |
| `HiddenDimension` | Gets or sets the hidden dimension of the transformer layers. |
| `NumGranularities` | Gets or sets the number of temporal granularity levels for guidance. |
| `NumHeads` | Gets or sets the number of attention heads. |
| `NumLayers` | Gets or sets the number of transformer layers. |

