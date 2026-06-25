---
title: "SiDModel<T>"
description: "Score Identity Distillation (SiD) model for single-step generation via score identity."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Diffusion.FastGeneration`

Score Identity Distillation (SiD) model for single-step generation via score identity.

## For Beginners

SiD uses a clever mathematical identity to train a fast model.
The pretrained diffusion model tells the student model exactly how its outputs differ
from real images, providing a precise training signal. This produces clean single-step
images without needing a separate discriminator network.

## How It Works

SiD distills a pretrained diffusion model into a single-step generator by exploiting
the score identity: the score function of the generator's output distribution should
match the pretrained model's score. This avoids the need for a discriminator or
regression loss, using only the pretrained score function for training.

Reference: Zhou et al., "Score Identity Distillation: Exponentially Fast Distillation of Pretrained Diffusion Models", ICML 2024

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `SiDModel(NeuralNetworkArchitecture<>,DiffusionModelOptions<>,INoiseScheduler<>,UNetNoisePredictor<>,StandardVAE<>,IConditioningModule<>,Nullable<Int32>)` | Initializes a new SiD model. |

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

