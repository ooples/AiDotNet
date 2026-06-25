---
title: "CCDMOptions<T>"
description: "Configuration options for CCDM (Conditional Continuous Diffusion Model for Time Series)."
section: "API Reference"
---

`Options & Configuration` · `AiDotNet.Models.Options`

Configuration options for CCDM (Conditional Continuous Diffusion Model for Time Series).

## For Beginners

CCDM is a diffusion-based forecasting model that:

**What is Diffusion?**
Diffusion models work by learning to remove noise. During training, noise is
progressively added to the target series. During inference, the model starts
from pure noise and iteratively denoises it, conditioned on the historical
context, to produce a forecast.

**Key Advantages:**

- Produces probabilistic forecasts (uncertainty estimates) naturally
- Operates in continuous space (no quantization loss)
- Score-matching objective is stable to train

**Trade-offs:**

- Slower inference than direct methods (requires multiple denoising steps)
- More parameters to tune (noise schedule, diffusion steps)

## How It Works

CCDM extends continuous diffusion models for conditional time series generation.
It operates in continuous space (unlike discrete token-based approaches) and uses
a score-matching objective for high-quality probabilistic forecasting.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `CCDMOptions` | Initializes a new instance with default values. |
| `CCDMOptions(CCDMOptions<>)` | Initializes a new instance by copying from another instance. |

## Properties

| Property | Summary |
|:-----|:--------|
| `BetaEnd` | Gets or sets the ending beta value for the linear noise schedule. |
| `BetaStart` | Gets or sets the starting beta value for the linear noise schedule. |
| `ContextLength` | Gets or sets the number of historical time steps used as input context. |
| `DiffusionSteps` | Gets or sets the number of diffusion (denoising) steps. |
| `DropoutRate` | Gets or sets the dropout rate for regularization. |
| `ForecastHorizon` | Gets or sets the number of future time steps to forecast. |
| `HiddenDimension` | Gets or sets the hidden dimension of the transformer layers. |
| `NumHeads` | Gets or sets the number of attention heads. |
| `NumLayers` | Gets or sets the number of transformer layers. |
| `SigmaMax` | Gets or sets the maximum noise level for the continuous diffusion schedule. |
| `SigmaMin` | Gets or sets the minimum noise level for the continuous diffusion schedule. |

