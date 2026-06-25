---
title: "Imagen3Model<T>"
description: "Imagen 3 model for text-to-image generation by Google DeepMind."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Diffusion.TextToImage`

Imagen 3 model for text-to-image generation by Google DeepMind.

## For Beginners

Imagen 3 is Google DeepMind's latest image generation model.

How Imagen 3 works:

1. Text is encoded by a Gemma text encoder (evolved from T5-XXL)
2. A base SiT (Scalable Interpolant Transformer) generates a 64x64 latent
3. A super-resolution cascade upscales to 256x256, then 1024x1024
4. Each stage is a separate diffusion model operating at increasing resolution
5. Human preference alignment ensures quality and safety

Key characteristics:

- Cascaded architecture: base (64x64) + SR (256x256) + SR (1024x1024)
- Gemma text encoder for deep prompt understanding
- Human feedback alignment (RLHF-style) for quality and safety
- SiT-based backbone with interpolant formulation
- 16 latent channels

Advantages:

- Exceptional prompt adherence and photorealism
- Strong safety filtering via RLHF alignment
- Excellent text rendering capabilities
- Cascaded approach enables very high resolution

Limitations:

- API-only (not open-source)
- Cascaded pipeline is slower than single-stage models
- Higher compute requirements due to multiple stages

## How It Works

Imagen 3 uses a cascaded diffusion approach with a base model and super-resolution stages.
It features Gemma-based text understanding (evolved from T5) and is aligned with human
feedback through RLHF-style training for improved safety and aesthetic quality.

Technical specifications:

- Architecture: Cascaded SiT with super-resolution stages
- Base: ~2B params, SiT backbone, 64x64 latent generation
- SR Stage 1: 256x256 upscale
- SR Stage 2: 1024x1024 final output
- Text encoder: Gemma (2048-dim embeddings)
- Latent channels: 16
- Default: 50 steps, guidance scale 7.5
- Resolution: 1024x1024 default, up to 2048x2048

Reference: Google DeepMind, "Imagen 3", 2024

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `Imagen3Model(NeuralNetworkArchitecture<>,DiffusionModelOptions<>,INoiseScheduler<>,SiTPredictor<>,StandardVAE<>,IConditioningModule<>,Nullable<Int32>)` | Initializes a new instance of Imagen3Model. |

## Properties

| Property | Summary |
|:-----|:--------|
| `Conditioner` |  |
| `LatentChannels` |  |
| `NoisePredictor` |  |
| `ParameterCount` |  |
| `UsesCascadedArchitecture` | Gets whether this model uses a cascaded architecture with super-resolution stages. |
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
| `SetParameters(Vector<>)` |  |

## Fields

| Field | Summary |
|:-----|:--------|
| `DefaultHeight` | Default image height for Imagen 3 (1024x1024). |
| `DefaultWidth` | Default image width for Imagen 3 (1024x1024). |

