---
title: "MetaDiffOptions<T, TInput, TOutput>"
description: "Configuration options for MetaDiff: Meta-Learning with Conditional Diffusion for Few-Shot Learning (Zhang et al., AAAI 2024)."
section: "API Reference"
---

`Options & Configuration` · `AiDotNet.MetaLearning.Options`

Configuration options for MetaDiff: Meta-Learning with Conditional Diffusion for Few-Shot
Learning (Zhang et al., AAAI 2024).

## How It Works

MetaDiff reframes the inner-loop gradient descent of meta-learning as a reverse diffusion
process over model weights. A task-conditional denoising network iteratively removes noise
to produce task-specific weights, conditioned on support-set features.

## Properties

| Property | Summary |
|:-----|:--------|
| `BetaEnd` | Ending value of the linear noise schedule β_T. |
| `BetaStart` | Starting value of the linear noise schedule β_1. |
| `DiffusionSteps` | Number of diffusion timesteps for the full forward/reverse process. |
| `SamplingSteps` | Number of denoising steps used during inference (≤ DiffusionSteps). |
| `TaskConditionDim` | Dimensionality of the task conditioning vector computed from support features. |

