---
title: "SDXLLightningModel<T>"
description: "SDXL Lightning model for 2-8 step high-quality generation via progressive distillation."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Diffusion.FastGeneration`

SDXL Lightning model for 2-8 step high-quality generation via progressive distillation.

## For Beginners

SDXL Lightning is ByteDance's answer to SDXL Turbo. It generates
beautiful 1024x1024 images in 2-8 steps (vs SDXL's 25-50). Unlike Turbo which works
best at 1 step, Lightning excels at 2-4 steps with slightly higher quality. It also
comes as a LoRA, so you can apply it to existing SDXL fine-tunes.

## How It Works

SDXL Lightning combines progressive distillation with adversarial training for high-quality
few-step generation. Available as both full UNet checkpoints and LoRA adapters. Achieves
superior quality to SDXL Turbo at 2-4 steps while maintaining SDXL-level aesthetics.

Reference: Lin et al., "SDXL-Lightning: Progressive Adversarial Diffusion Distillation", 2024

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `SDXLLightningModel(NeuralNetworkArchitecture<>,DiffusionModelOptions<>,INoiseScheduler<>,UNetNoisePredictor<>,StandardVAE<>,IConditioningModule<>,Nullable<Int32>)` | Initializes a new SDXL Lightning model. |

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

