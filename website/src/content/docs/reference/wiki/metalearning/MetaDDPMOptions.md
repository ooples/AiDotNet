---
title: "MetaDDPMOptions<T, TInput, TOutput>"
description: "Configuration options for Meta-DDPM: Meta-Learning with Denoising Diffusion Probabilistic Models."
section: "API Reference"
---

`Options & Configuration` · `AiDotNet.MetaLearning.Options`

Configuration options for Meta-DDPM: Meta-Learning with Denoising Diffusion Probabilistic Models.

## How It Works

Meta-DDPM extends the DDPM framework to meta-learning by meta-learning a noise prediction
network that generates task-specific model weights conditioned on support set embeddings.
Unlike MetaDiff (which models weight deltas), Meta-DDPM directly generates the full
adapted weight vector using the DDPM generative framework with a learned linear noise
schedule and task-conditional denoising.

## Properties

| Property | Summary |
|:-----|:--------|
| `BetaEnd` | Ending beta for the linear noise schedule. |
| `BetaStart` | Starting beta for the linear noise schedule. |
| `EmaDecay` | Weight for EMA (exponential moving average) of model parameters for stable generation. |
| `NumTimesteps` | Total timesteps in the DDPM diffusion process. |
| `SamplingSteps` | Number of denoising steps used during generation (can be less than NumTimesteps). |
| `TaskConditionDim` | Dimensionality of the task conditioning vector. |

