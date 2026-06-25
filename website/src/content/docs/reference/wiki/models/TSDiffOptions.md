---
title: "TSDiffOptions<T>"
description: "Configuration options for TSDiff (Time Series Diffusion for unconditional/conditional generation)."
section: "API Reference"
---

`Options & Configuration` · `AiDotNet.Models.Options`

Configuration options for TSDiff (Time Series Diffusion for unconditional/conditional generation).

## For Beginners

TSDiff is designed as a versatile time series generator
that can handle multiple tasks with one architecture.

**The Key Insight:**
Different time series tasks (forecasting, imputation, generation) can all be viewed
as conditional generation problems. TSDiff uses a unified framework with different
conditioning strategies:

**Supported Tasks:**

1. **Unconditional Generation:** Generate synthetic time series from scratch
2. **Forecasting:** Condition on historical data to predict future
3. **Imputation:** Condition on observed values to fill missing
4. **Refinement:** Condition on noisy data to produce clean version

**TSDiff Architecture:**

- Self-guided diffusion: Uses attention over time for temporal coherence
- Observation guidance: Gradient-based conditioning on observations
- Flexible scheduler: Different noise schedules for different tasks
- Multi-resolution: Captures patterns at multiple time scales

**Key Benefits:**

- Single model for multiple tasks
- Can combine conditioning strategies
- Generates long, coherent sequences
- Captures complex temporal dynamics

## How It Works

TSDiff is a flexible diffusion model for time series that supports both unconditional
generation and various conditioning mechanisms for forecasting and imputation.

**Reference:** Kollovieh et al., "Predict, Refine, Synthesize: Self-Guiding Diffusion Models for Probabilistic Time Series Forecasting", 2023.
https://arxiv.org/abs/2307.11494

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `TSDiffOptions` | Initializes a new instance of the `TSDiffOptions` class with default values. |
| `TSDiffOptions(TSDiffOptions<>)` | Initializes a new instance by copying from another instance. |

## Properties

| Property | Summary |
|:-----|:--------|
| `BetaEnd` | Gets or sets the ending noise level (beta_T). |
| `BetaSchedule` | Gets or sets the noise schedule type. |
| `BetaStart` | Gets or sets the starting noise level (beta_1). |
| `DropoutRate` | Gets or sets the dropout rate for regularization. |
| `ForecastHorizon` | Gets or sets the forecast horizon (prediction length). |
| `GuidanceScale` | Gets or sets the guidance scale for classifier-free guidance. |
| `HiddenDimension` | Gets or sets the hidden dimension for the denoising network. |
| `KernelSize` | Gets or sets the kernel size for temporal convolutions. |
| `NumAttentionHeads` | Gets or sets the number of attention heads in self-attention layers. |
| `NumDiffusionSteps` | Gets or sets the number of diffusion steps. |
| `NumFeatures` | Gets or sets the number of features (variables). |
| `NumResidualBlocks` | Gets or sets the number of residual blocks in the denoising network. |
| `NumSamples` | Gets or sets the number of samples for uncertainty estimation. |
| `SequenceLength` | Gets or sets the sequence length (context + forecast). |
| `UseObservationGuidance` | Gets or sets whether to use observation guidance for conditioning. |
| `UseSelfGuidance` | Gets or sets whether to use self-guidance during sampling. |

