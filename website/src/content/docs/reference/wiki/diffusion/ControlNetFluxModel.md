---
title: "ControlNetFluxModel<T>"
description: "ControlNet adapted for the FLUX.1 architecture with flow matching."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Diffusion.Control`

ControlNet adapted for the FLUX.1 architecture with flow matching.

## For Beginners

This brings ControlNet's "follow my reference image" capability
to FLUX models. FLUX uses a different internal architecture than Stable Diffusion,
so this specialized version ensures control signals work correctly with FLUX.

## How It Works

Adapts ControlNet conditioning for FLUX.1's flow-matching diffusion framework
with double-stream transformer blocks. Uses 16-channel latent space and
rectified flow scheduling native to FLUX.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `ControlNetFluxModel(NeuralNetworkArchitecture<>,DiffusionModelOptions<>,INoiseScheduler<>,FluxDoubleStreamPredictor<>,StandardVAE<>,IConditioningModule<>,ControlType,Nullable<Int32>)` | Initializes a new ControlNet-FLUX model. |

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

