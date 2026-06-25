---
title: "MultistepLCModel<T>"
description: "Multistep Latent Consistency Model (MLCM) for high-quality few-step generation."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Diffusion.FastGeneration`

Multistep Latent Consistency Model (MLCM) for high-quality few-step generation.

## For Beginners

LCM generates images in 2-4 steps. MLCM improves on this by
training the model to be consistent across different numbers of steps — meaning it
produces good results whether you use 2, 4, or 8 steps. This flexibility lets you
choose the best speed/quality tradeoff for your use case.

## How It Works

MLCM extends Latent Consistency Models to support multiple distillation targets
per training step, improving generation quality at 2-8 steps. Uses a multistep
consistency loss that enforces the model to be self-consistent across multiple
denoising trajectories simultaneously.

Reference: Based on Latent Consistency Models (Luo et al., 2023) with multistep extensions

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `MultistepLCModel(NeuralNetworkArchitecture<>,DiffusionModelOptions<>,INoiseScheduler<>,UNetNoisePredictor<>,StandardVAE<>,IConditioningModule<>,Nullable<Int32>)` | Initializes a new Multistep Latent Consistency Model. |

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

