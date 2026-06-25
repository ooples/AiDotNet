---
title: "SDTurboModel<T>"
description: "SD Turbo / SDXL Turbo model for real-time single-step image generation."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Diffusion.FastGeneration`

SD Turbo / SDXL Turbo model for real-time single-step image generation.

## For Beginners

SD Turbo generates images almost instantly:

How SD/SDXL Turbo works:

1. Uses the same architecture as SD 1.5 / SDXL
2. Trained with Adversarial Diffusion Distillation (ADD)
3. A discriminator network enforces realism at each step
4. Can generate high-quality images in just 1-4 denoising steps

Key characteristics:

- SD Turbo: 512x512, based on SD 2.1 architecture
- SDXL Turbo: 512x512, based on SDXL architecture
- 1-4 steps instead of 20-50 (10-50x faster)
- No classifier-free guidance needed (guidance scale = 0)
- Uses ADD (Adversarial Diffusion Distillation) training

Advantages:

- Near real-time generation (~0.1 seconds)
- Single-step generation possible
- Same quality as multi-step at much lower latency

Limitations:

- Lower diversity than full multi-step models
- Less control via guidance scale (guidance=0 recommended)
- Smaller effective resolution than full SDXL
- Non-commercial license for Turbo models

## How It Works

SD Turbo and SDXL Turbo are distilled versions of Stable Diffusion and SDXL
that can generate images in 1-4 steps using Adversarial Diffusion Distillation (ADD).

Technical specifications:

- Architecture: Distilled SD 2.1 / SDXL via Adversarial Diffusion Distillation
- SD Turbo: SD 2.1 U-Net (865M params), 1024-dim cross-attention
- SDXL Turbo: SDXL U-Net (2.6B params), dual text encoders
- Steps: 1-4 (optimal: 1 for speed, 4 for quality)
- Guidance scale: 0.0 (no CFG needed)
- Resolution: 512x512 (both variants)
- Distillation: ADD with adversarial loss + diffusion loss

Reference: Sauer et al., "Adversarial Diffusion Distillation", 2023

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `SDTurboModel(NeuralNetworkArchitecture<>,DiffusionModelOptions<>,INoiseScheduler<>,UNetNoisePredictor<>,StandardVAE<>,IConditioningModule<>,Boolean,Nullable<Int32>)` | Initializes a new instance of SDTurboModel with full customization support. |

## Properties

| Property | Summary |
|:-----|:--------|
| `Conditioner` |  |
| `IsXLVariant` | Gets whether this is the SDXL Turbo variant (true) or SD Turbo variant (false). |
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
| `ImageToImage(Tensor<>,String,String,Double,Int32,Nullable<Double>,Nullable<Int32>)` |  |
| `InitializeLayers(UNetNoisePredictor<>,StandardVAE<>,Nullable<Int32>)` | Initializes the distilled U-Net and VAE layers, using custom layers from the user if provided or creating industry-standard layers from the ADD paper. |
| `SetParameters(Vector<>)` |  |

## Fields

| Field | Summary |
|:-----|:--------|
| `DefaultHeight` | Default image height for SD Turbo generation. |
| `DefaultWidth` | Default image width for SD Turbo generation. |
| `TURBO_CROSS_ATTENTION_DIM` | Cross-attention dimension (1024 for SD Turbo based on SD 2.1). |
| `TURBO_DEFAULT_GUIDANCE_SCALE` | Default guidance scale for Turbo models (0.0 = no CFG, as recommended). |

