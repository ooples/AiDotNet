---
title: "HiDreamModel<T>"
description: "HiDream-I1 model for high-quality imaginative text-to-image generation."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Diffusion.TextToImage`

HiDream-I1 model for high-quality imaginative text-to-image generation.

## For Beginners

HiDream is designed to generate more creative and imaginative
images with excellent prompt understanding.

How HiDream works:

1. Text is encoded by CLIP ViT-L/14 and Llama-3.1 (a powerful language model)
2. An MMDiT-X transformer processes text and image tokens with joint attention
3. Flow matching training enables efficient 28-50 step generation
4. A 16-channel VAE decodes latents to high-resolution images

Model variants:

- HiDream-I1 Full: 17B parameters, highest quality
- HiDream-I1 Dev: 12B parameters, good balance of quality and speed
- HiDream-I1 Fast: 8B parameters, optimized for speed

Key characteristics:

- Llama-3.1 as text encoder for deep language understanding
- MMDiT-X architecture with enhanced cross-attention
- 16 latent channels
- Flow matching training
- Excellent at artistic and fantasy-style content
- Strong composition and spatial reasoning

Advantages:

- Superior prompt understanding via Llama-3.1
- Great artistic and creative generation
- Multiple speed/quality tradeoff variants
- Open-weight model

Limitations:

- Full variant requires significant VRAM (~40GB)
- Newer model with smaller community
- Llama-3.1 encoder adds overhead

## How It Works

HiDream-I1 uses an MMDiT architecture enhanced with Llama-3.1 as a text encoder,
providing superior prompt understanding through a large language model backbone.
It features improved composition handling and artistic style diversity compared to
standard diffusion models, with variants ranging from 8B to 17B parameters.

Technical specifications:

- Architecture: MMDiT-X with Llama-3.1 text conditioning
- Full: 17B params, hidden 4096, 40 layers
- Dev: 12B params, hidden 3072, 32 layers
- Fast: 8B params, hidden 2560, 24 layers
- Text encoder 1: CLIP ViT-L/14 (768-dim, pooled)
- Text encoder 2: Llama-3.1-8B (4096-dim, sequence)
- Patch size: 2
- VAE: 16 latent channels, 8x spatial compression
- Training: Flow matching
- Resolution: 1024x1024 default

Reference: HiDream.ai, "HiDream-I1: High-Quality Imaginative Image Generation", 2025

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `HiDreamModel(NeuralNetworkArchitecture<>,DiffusionModelOptions<>,INoiseScheduler<>,MMDiTXNoisePredictor<>,StandardVAE<>,IConditioningModule<>,HiDreamVariant,Nullable<Int32>)` | Initializes a new instance of HiDreamModel. |

## Properties

| Property | Summary |
|:-----|:--------|
| `Conditioner` |  |
| `LatentChannels` |  |
| `NoisePredictor` |  |
| `ParameterCount` |  |
| `VAE` |  |
| `Variant` | Gets the model variant. |

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
| `DefaultHeight` | Default image height for HiDream (1024x1024). |
| `DefaultWidth` | Default image width for HiDream (1024x1024). |

