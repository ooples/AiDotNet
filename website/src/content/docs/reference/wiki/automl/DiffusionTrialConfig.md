---
title: "DiffusionTrialConfig<T>"
description: "Configuration for a diffusion model trial in AutoML."
section: "API Reference"
---

`Options & Configuration` · `AiDotNet.AutoML`

Configuration for a diffusion model trial in AutoML.

## Properties

| Property | Summary |
|:-----|:--------|
| `BaseChannels` | Gets or sets the base channels for the noise predictor. |
| `GuidanceScale` | Gets or sets the guidance scale for classifier-free guidance. |
| `InferenceSteps` | Gets or sets the number of inference steps. |
| `LatentDim` | Gets or sets the latent dimension (channels). |
| `LatentHeight` | Gets or sets the latent spatial height (default: 64 for 512x512 images with 8x downscaling). |
| `LatentWidth` | Gets or sets the latent spatial width (default: 64 for 512x512 images with 8x downscaling). |
| `LearningRate` | Gets or sets the learning rate for training. |
| `NoisePredictorType` | Gets or sets the noise predictor type. |
| `NumResBlocks` | Gets or sets the number of residual blocks per level. |
| `SchedulerType` | Gets or sets the scheduler type. |
| `Seed` | Gets or sets the optional random seed. |

## Methods

| Method | Summary |
|:-----|:--------|
| `FromDictionary(Dictionary<String,Object>)` | Creates a configuration from a dictionary of parameters. |
| `ToDictionary` | Converts the configuration to a dictionary of parameters. |

