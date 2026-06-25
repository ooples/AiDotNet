---
title: "TCDModel<T>"
description: "Trajectory Consistency Distillation (TCD) model for high-quality few-step generation."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Diffusion.FastGeneration`

Trajectory Consistency Distillation (TCD) model for high-quality few-step generation.

## For Beginners

TCD improves upon LCM by being smarter about how it trains.
While LCM focuses on individual steps, TCD considers the entire journey from noise
to image. Adding a tiny bit of random noise during generation also helps — like how
a slight shake of the camera can sometimes produce a more natural photo.

## How It Works

TCD extends LCM by using a trajectory-aware loss that considers the entire denoising
path, not just individual timestep consistency. Combined with a stochastic noise
injection strategy during inference, TCD produces higher-quality images than LCM
at the same step count, especially at 2-4 steps.

Reference: Zheng et al., "Trajectory Consistency Distillation", 2024

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `TCDModel(NeuralNetworkArchitecture<>,DiffusionModelOptions<>,INoiseScheduler<>,UNetNoisePredictor<>,StandardVAE<>,IConditioningModule<>,Nullable<Int32>)` | Initializes a new TCD model. |

## Properties

| Property | Summary |
|:-----|:--------|
| `Conditioner` |  |
| `LatentChannels` |  |
| `NoisePredictor` |  |
| `ParameterCount` |  |
| `VAE` |  |

## Methods

| Method | Summary |
|:-----|:--------|
| `Clone` |  |
| `DeepCopy` |  |
| `GetModelMetadata` |  |
| `GetParameters` |  |
| `SetParameters(Vector<>)` |  |

## Fields

| Field | Summary |
|:-----|:--------|
| `OPTIMAL_INFERENCE_STEPS` | Paper-optimal sampling step count. |

