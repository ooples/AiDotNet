---
title: "PixArtDeltaLCMModel<T>"
description: "PixArt-Delta LCM for few-step generation from the efficient PixArt-Delta DiT architecture."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Diffusion.FastGeneration`

PixArt-Delta LCM for few-step generation from the efficient PixArt-Delta DiT architecture.

## For Beginners

PixArt-Delta is an efficient alternative to SD3 — it produces
great images with much less training data and compute. This LCM version makes it
even faster at inference time, generating in just 2-8 steps. Good for when you want
high quality but don't need FLUX/SD3-level capability.

## How It Works

Applies Latent Consistency Model distillation to PixArt-Delta, a training-efficient
DiT model. PixArt-Delta uses a decomposed cross-attention mechanism that significantly
reduces training cost. The LCM variant enables 2-8 step generation while maintaining
PixArt-Delta's strong text-image alignment.

Reference: Chen et al., "PixArt-delta: Fast and Controllable Image Generation with Latent Consistency Models", 2024

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

