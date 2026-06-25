---
title: "LuminaImage2Model<T>"
description: "Lumina Image 2.0 model for high-resolution text-to-image generation."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Diffusion.TextToImage`

Lumina Image 2.0 model for high-resolution text-to-image generation.

## For Beginners

Lumina Image 2.0 is an open-source model from the Lumina framework.

How Lumina Image 2.0 works:

1. Text is encoded by Gemma 2B for strong multilingual understanding
2. A Flag-DiT (Flow-Aware Generative DiT) processes tokens with flow matching
3. Multi-resolution support handles different aspect ratios natively
4. A 16-channel VAE decodes latents to high-resolution images

Key characteristics:

- Flag-DiT architecture with flow-aware generation
- Gemma 2B text encoder
- Native multi-resolution and aspect ratio support
- 2B parameters in the transformer
- 16 latent channels
- Up to 2K resolution generation

Advantages:

- Open-source with permissive license
- Native multi-resolution support
- Good quality-to-compute ratio
- Flexible aspect ratio handling

## How It Works

Lumina Image 2.0 uses a Flag-DiT (Flow-Aware Generative DiT) architecture with improved
multi-resolution support and efficient attention mechanisms. It features Gemma text encoding
and flow matching training for generating images up to 2K resolution with excellent detail.

Reference: Gao et al., "Lumina-Image: High-Resolution Image Generation with
Flow-Aware Generative Transformers", 2024

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
| `ImageToImage(Tensor<>,String,String,Double,Int32,Nullable<Double>,Nullable<Int32>)` |  |
| `SetParameters(Vector<>)` |  |

