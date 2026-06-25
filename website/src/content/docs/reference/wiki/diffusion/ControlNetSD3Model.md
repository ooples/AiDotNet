---
title: "ControlNetSD3Model<T>"
description: "ControlNet adapted for Stable Diffusion 3's MMDiT architecture."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Diffusion.Control`

ControlNet adapted for Stable Diffusion 3's MMDiT architecture.

## For Beginners

This brings ControlNet control to Stable Diffusion 3 models.
SD3 uses a completely different architecture from SD1.5/SDXL, so this version
is specially designed to inject control signals into the transformer blocks.

## How It Works

Adapts ControlNet conditioning for SD3's Multi-Modal Diffusion Transformer (MMDiT)
architecture. Uses 16-channel latent space and supports dual text encoders
(CLIP + T5) for enhanced prompt understanding with control signal injection.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `ControlNetSD3Model(NeuralNetworkArchitecture<>,DiffusionModelOptions<>,INoiseScheduler<>,MMDiTXNoisePredictor<>,StandardVAE<>,IConditioningModule<>,ControlType,Nullable<Int32>)` | Initializes a new ControlNet-SD3 model. |

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

