---
title: "DiffusionModelOptions<T>"
description: "Configuration options for diffusion-based generative models."
section: "API Reference"
---

`Options & Configuration` · `AiDotNet.Models.Options`

Configuration options for diffusion-based generative models.

## For Beginners

Diffusion models work by learning to reverse a gradual noising process.
These options control how the model trains and generates samples:

- LearningRate: How big of a step to take during training
- TrainTimesteps: How many noise levels to use (more = finer control)
- BetaStart/BetaEnd: How much noise at each step

## How It Works

This options class provides configuration for all diffusion model parameters including
training hyperparameters, scheduler configuration, and generation settings.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `DiffusionModelOptions` | Initializes a new instance with default values. |
| `DiffusionModelOptions(DiffusionModelOptions<>)` | Initializes a new instance by copying from another instance. |

## Properties

| Property | Summary |
|:-----|:--------|
| `BetaEnd` | Gets or sets the ending beta value (noise variance at t=T). |
| `BetaSchedule` | Gets or sets the type of beta schedule to use. |
| `BetaStart` | Gets or sets the starting beta value (noise variance at t=0). |
| `ClipSample` | Gets or sets whether to clip predicted samples to [-1, 1]. |
| `DefaultInferenceSteps` | Gets or sets the default number of inference steps for generation. |
| `LearningRate` | Gets or sets the learning rate for training parameter updates. |
| `LossFunction` | Gets or sets the loss function for training. |
| `PredictionType` | Gets or sets what the model is trained to predict. |
| `TrainTimesteps` | Gets or sets the number of timesteps used during training. |
| `UseGpuExecutionGraph` | Gets or sets whether each synchronous denoising-step noise prediction runs inside a GPU deferred execution graph (device-resident, fused, multi-stream) instead of eager per-op dispatch. |

