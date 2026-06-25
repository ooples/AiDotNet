---
title: "HyperSDModel<T>"
description: "Hyper-SD model for unified 1-8 step generation via trajectory-segmented distillation."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Diffusion.FastGeneration`

Hyper-SD model for unified 1-8 step generation via trajectory-segmented distillation.

## For Beginners

Most fast-generation models are optimized for a specific step
count (1-step or 4-step). Hyper-SD works well across ALL step counts from 1 to 8,
letting you smoothly trade speed for quality. It's like having multiple specialized
models in one — just change the step count.

## How It Works

Hyper-SD uses a novel trajectory-segmented consistency distillation that divides the
denoising trajectory into segments and distills each segment independently. Combined
with human feedback learning, it achieves state-of-the-art quality across 1-8 step
configurations for both SD 1.5 and SDXL architectures.

Reference: Ren et al., "Hyper-SD: Trajectory Segmented Consistency Model", 2024

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `HyperSDModel(NeuralNetworkArchitecture<>,DiffusionModelOptions<>,INoiseScheduler<>,UNetNoisePredictor<>,StandardVAE<>,IConditioningModule<>,Boolean,Nullable<Int32>)` | Initializes a new Hyper-SD model. |

## Properties

| Property | Summary |
|:-----|:--------|
| `Conditioner` |  |
| `IsXLVariant` | Gets whether this is the SDXL variant. |
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

