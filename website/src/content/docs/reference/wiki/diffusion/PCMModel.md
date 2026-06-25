---
title: "PCMModel<T>"
description: "Phased Consistency Model (PCM) for flexible-step generation with phase-based training."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Diffusion.FastGeneration`

Phased Consistency Model (PCM) for flexible-step generation with phase-based training.

## For Beginners

Standard consistency models work great at 1-2 steps but lose
quality at higher steps. PCM fixes this by dividing the generation process into phases
and handling each phase separately. This means you get good results from 1 to 16 steps,
giving you maximum flexibility in the speed/quality tradeoff.

## How It Works

PCM divides the diffusion trajectory into phases and enforces consistency within each
phase rather than across the entire trajectory. This phased approach allows the model
to maintain high quality across different step configurations (1, 2, 4, 8, 16 steps)
without the quality degradation seen in standard consistency models at higher steps.

Reference: Wang et al., "Phased Consistency Model", NeurIPS 2024

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `PCMModel(NeuralNetworkArchitecture<>,DiffusionModelOptions<>,INoiseScheduler<>,UNetNoisePredictor<>,StandardVAE<>,IConditioningModule<>,Boolean,Nullable<Int32>)` | Initializes a new Phased Consistency Model. |

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

