---
title: "PixArtDeltaModel<T>"
description: "PixArt-Delta model — LCM-distilled PixArt for fast 2-8 step generation."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Diffusion.TextToImage`

PixArt-Delta model — LCM-distilled PixArt for fast 2-8 step generation.

## For Beginners

PixArt-Delta is PixArt-Alpha made faster:

Key characteristics:

- Same DiT architecture as PixArt-Alpha
- LCM distillation: trained to produce good results in 2-8 steps
- No classifier-free guidance needed (saves 50% compute)
- Maintains T5-XXL text encoder for good prompt understanding

How it compares:

- PixArt-Alpha: ~20 steps, needs CFG → slower
- PixArt-Delta: ~4 steps, no CFG needed → 10x faster
- SD Turbo: 1-4 steps but lower quality
- PixArt-Delta: 2-8 steps with higher quality

Use PixArt-Delta when you need:

- Fast generation (2-8 steps)
- Good quality without guidance overhead
- Resource-efficient deployment

## How It Works

PixArt-Delta applies Latent Consistency Model (LCM) distillation to PixArt-Alpha,
enabling high-quality image generation in just 2-8 denoising steps while preserving
the efficient DiT architecture.

Technical specifications:

- Architecture: DiT-XL/2 (same as PixArt-Alpha)
- Distillation: Latent Consistency Model (LCM)
- Steps: 2-8 (optimal: 4)
- Guidance scale: 1.0 (no CFG)
- Text encoder: T5-XXL (4096-dim)
- Resolution: 512x512 to 1024x1024

Reference: Chen et al., "PixArt-delta: Fast and Controllable Image Generation
with Latent Consistency Models", 2024

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `PixArtDeltaModel(NeuralNetworkArchitecture<>,DiffusionModelOptions<>,INoiseScheduler<>,DiTNoisePredictor<>,StandardVAE<>,IConditioningModule<>,Nullable<Int32>)` | Initializes a new instance of PixArtDeltaModel with full customization support. |

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
| `GenerateFromText(String,String,Int32,Int32,Int32,Nullable<Double>,Nullable<Int32>)` |  |
| `GetModelMetadata` |  |
| `GetParameters` |  |
| `SetParameters(Vector<>)` |  |

