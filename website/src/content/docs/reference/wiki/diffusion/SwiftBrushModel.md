---
title: "SwiftBrushModel<T>"
description: "SwiftBrush model for image-free one-step text-to-image distillation."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Diffusion.FastGeneration`

SwiftBrush model for image-free one-step text-to-image distillation.

## For Beginners

Most distillation methods need real images to train the fast
model. SwiftBrush skips this entirely — it trains using only the teacher model's
knowledge, like a student learning to draw by watching the teacher correct their
work, without needing reference images. The result is a one-step generator.

## How It Works

SwiftBrush distills a diffusion model into a single-step generator without using any
real images during training. The student generates a latent, which the frozen teacher
denoises to provide a training signal. This "image-free" approach avoids dataset
licensing issues and simplifies the training pipeline.

Reference: Nguyen et al., "SwiftBrush: One-Step Text-to-Image Diffusion Model with Variational Score Distillation", CVPR 2024

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `SwiftBrushModel(NeuralNetworkArchitecture<>,DiffusionModelOptions<>,INoiseScheduler<>,UNetNoisePredictor<>,StandardVAE<>,IConditioningModule<>,Nullable<Int32>)` | Initializes a new SwiftBrush model. |

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

