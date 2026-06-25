---
title: "ImprovedConsistencyModel<T>"
description: "Improved Consistency Training (iCT) model for single-step image generation."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Diffusion.FastGeneration`

Improved Consistency Training (iCT) model for single-step image generation.

## For Beginners

Consistency models learn to map any noisy image directly to the
clean image in a single step. Unlike diffusion models that remove noise gradually over
20-50 steps, iCT does it in one shot. The "improved" version trains better by using
smarter loss functions and noise schedules, producing higher quality single-step images.

## How It Works

iCT improves upon the original Consistency Training by using a lognormal schedule for
the noise discretization, pseudo-Huber loss instead of L2, and an improved EMA schedule.
Achieves state-of-the-art FID scores among single-step generators on ImageNet 64x64.

Reference: Song and Dhariwal, "Improved Techniques for Training Consistency Models", ICLR 2024

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `ImprovedConsistencyModel(NeuralNetworkArchitecture<>,DiffusionModelOptions<>,INoiseScheduler<>,UNetNoisePredictor<>,StandardVAE<>,IConditioningModule<>,Nullable<Int32>)` | Initializes a new Improved Consistency Training model. |

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

