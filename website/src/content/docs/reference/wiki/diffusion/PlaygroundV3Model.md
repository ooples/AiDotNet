---
title: "PlaygroundV3Model<T>"
description: "Playground v3 model for aesthetically optimized text-to-image generation."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Diffusion.TextToImage`

Playground v3 model for aesthetically optimized text-to-image generation.

## For Beginners

Playground v3 is specifically trained to generate beautiful,
aesthetically pleasing images.

How Playground v3 works:

1. Text is encoded by triple encoders: CLIP ViT-L, OpenCLIP ViT-bigG, and T5-XXL
2. An MMDiT-X transformer (same family as SD3.5) denoises the latent
3. Aesthetic reward training ensures outputs are visually appealing
4. EDM2-style preconditioning improves sampling efficiency

Key characteristics:

- MMDiT-X architecture optimized for aesthetics
- DPO (Direct Preference Optimization) training for visual appeal
- Triple text encoders for comprehensive prompt understanding
- EDM2-style preconditioning
- 16 latent channels
- 25 default inference steps

Advantages:

- Consistently generates aesthetically beautiful images
- Strong color harmony and composition
- Excellent skin tones and lighting in portraits
- Good balance of realism and artistic quality

Limitations:

- May sacrifice strict prompt adherence for aesthetics
- Aesthetic bias may not suit all use cases
- Limited stylistic diversity compared to unaligned models

## How It Works

Playground v3 is optimized for aesthetic quality through human preference training with
DPO (Direct Preference Optimization). It uses an MMDiT-X architecture similar to SD3.5
but fine-tuned extensively on human aesthetic preference data, producing images that
consistently score highly on visual appeal metrics.

Technical specifications:

- Architecture: MMDiT-X with aesthetic DPO alignment
- Backbone: ~8B params, hidden 4096, 38 layers, 64 heads
- Text encoder 1: CLIP ViT-L/14 (768-dim)
- Text encoder 2: OpenCLIP ViT-bigG (1280-dim)
- Text encoder 3: T5-XXL (4096-dim)
- Training: Rectified flow + DPO aesthetic alignment
- VAE: 16 latent channels, 8x spatial compression
- Default: 25 steps, guidance scale 5.0
- Resolution: 1024x1024 default

Reference: Li et al., "Playground v3: Improving Text-to-Image Alignment with
Human Feedback", 2024

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `PlaygroundV3Model(NeuralNetworkArchitecture<>,DiffusionModelOptions<>,INoiseScheduler<>,MMDiTXNoisePredictor<>,StandardVAE<>,IConditioningModule<>,Nullable<Int32>)` | Initializes a new instance of PlaygroundV3Model. |

## Properties

| Property | Summary |
|:-----|:--------|
| `Conditioner` |  |
| `LatentChannels` |  |
| `NoisePredictor` |  |
| `ParameterCount` |  |
| `UsesAestheticDPO` | Gets whether this model uses aesthetic DPO alignment. |
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
| `DefaultHeight` | Default image height for Playground v3 (1024x1024). |
| `DefaultWidth` | Default image width for Playground v3 (1024x1024). |

