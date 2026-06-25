---
title: "AuraFlowModel<T>"
description: "AuraFlow model — open-source flow-matching text-to-image model."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Diffusion.FastGeneration`

AuraFlow model — open-source flow-matching text-to-image model.

## For Beginners

AuraFlow is a community-driven alternative to commercial models:

Key characteristics:

- Flow matching: continuous-time formulation (not discrete DDPM steps)
- DiT backbone: transformer-based denoiser
- Open-source: fully open weights and code
- T5 text encoder for strong prompt understanding
- Competitive quality with commercial models

Use AuraFlow when you need:

- Open-source flow-matching model
- Alternative to SD3/Flux architecture
- Research-friendly model

## How It Works

AuraFlow is an open-source flow-matching model from Fal.ai that uses a modified
DiT architecture with flow matching for high-quality text-to-image generation.

Technical specifications:

- Architecture: DiT with flow matching
- Text encoder: T5-XXL (4096-dim)
- Resolution: 1024x1024
- Latent channels: 4, 8x downsampling
- Scheduler: Flow matching (Euler/midpoint)

Reference: Fal.ai, "AuraFlow v0.3", 2024

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

