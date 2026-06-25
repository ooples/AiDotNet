---
title: "SchedulerConfig<T>"
description: "Configuration options for diffusion model step schedulers."
section: "API Reference"
---

`Options & Configuration` · `AiDotNet.Diffusion.Schedulers`

Configuration options for diffusion model step schedulers.

## For Beginners

This is like a settings panel for the scheduler. You can control:

- How many steps to use (more = higher quality, slower)
- How much noise to start and end with (the beta values)
- What pattern of noise to use (linear, scaled, cosine)
- Whether to clip values to prevent extreme outputs
- What the model is predicting (noise, sample, or velocity)

The default values are research-backed and work well for most cases.

## How It Works

This configuration class defines all the parameters needed to initialize a step scheduler.
These parameters control the noise schedule and behavior of the diffusion process.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `SchedulerConfig(Int32,,,BetaSchedule,Boolean,DiffusionPredictionType)` | Initializes a new scheduler configuration. |

## Properties

| Property | Summary |
|:-----|:--------|
| `BetaEnd` | Gets the ending beta value (noise variance at t=T). |
| `BetaSchedule` | Gets the type of beta schedule to use. |
| `BetaStart` | Gets the starting beta value (noise variance at t=0). |
| `ClipSample` | Gets whether to clip predicted samples to [-1, 1]. |
| `PredictionType` | Gets the type of prediction the model makes. |
| `TrainTimesteps` | Gets the number of timesteps used during training. |

## Methods

| Method | Summary |
|:-----|:--------|
| `CreateDefault` | Creates a default configuration for DDPM-style models. |
| `CreateLCM` | Creates a configuration for LCM (Latent Consistency Model) sampling. |
| `CreateRectifiedFlow` | Creates a configuration for rectified flow models (SD3, FLUX.1). |
| `CreateStableDiffusion` | Creates a configuration optimized for Stable Diffusion-style models. |

