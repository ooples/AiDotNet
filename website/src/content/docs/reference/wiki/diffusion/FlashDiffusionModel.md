---
title: "FlashDiffusionModel<T>"
description: "Flash Diffusion model for rapid few-step generation via progressive attention distillation."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Diffusion.FastGeneration`

Flash Diffusion model for rapid few-step generation via progressive attention distillation.

## For Beginners

Most distillation methods match the final output of teacher and
student. Flash Diffusion goes deeper — it matches the internal "attention patterns"
(how the model decides which parts of the prompt affect which parts of the image).
This means the fast model better understands complex prompts like "a red car next to
a blue house" even in just 4 steps.

## How It Works

Flash Diffusion uses progressive attention distillation where the student learns to
reproduce the teacher's attention patterns at progressively fewer steps. This approach
preserves the compositional abilities of the original model better than pure output
distillation, maintaining text-image alignment at low step counts.

Reference: Chadebec et al., "Flash Diffusion: Accelerating Any Conditional Diffusion Model for Few Steps Image Generation", 2024

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `FlashDiffusionModel(NeuralNetworkArchitecture<>,DiffusionModelOptions<>,INoiseScheduler<>,UNetNoisePredictor<>,StandardVAE<>,IConditioningModule<>,Nullable<Int32>)` | Initializes a new Flash Diffusion model. |

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

