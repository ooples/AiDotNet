---
title: "KolorsModel<T>"
description: "Kolors model — ChatGLM3-powered bilingual text-to-image model by Kwai."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Diffusion.TextToImage`

Kolors model — ChatGLM3-powered bilingual text-to-image model by Kwai.

## For Beginners

Kolors is like SDXL but with Chinese language understanding:

Key characteristics:

- ChatGLM3-6B text encoder: strong Chinese + English understanding
- SDXL-like U-Net backbone: proven architecture for quality
- 4096-dim cross-attention from ChatGLM3 embeddings
- Native 1024x1024 resolution
- Open-source with Apache 2.0 license

How Kolors works:

1. Text goes through ChatGLM3-6B (6B parameter language model)
2. 4096-dim embeddings provide rich text understanding
3. SDXL-like U-Net denoises with cross-attention to embeddings
4. VAE decodes to final image

Use Kolors when you need:

- Chinese text-to-image generation
- Open-source bilingual model
- SDXL-quality with multilingual support

## How It Works

Kolors is Kwai's text-to-image model that uses ChatGLM3-6B as the text encoder,
providing strong bilingual (Chinese-English) text understanding. It builds on the
SDXL U-Net architecture with ChatGLM3's 4096-dim embeddings for cross-attention.

Technical specifications:

- Architecture: SDXL U-Net with ChatGLM3-6B text encoder
- Text encoder: ChatGLM3-6B (4096-dim, 65024 vocab)
- U-Net: ~2.6B parameters
- Resolution: 1024x1024
- Latent space: 4 channels, 8x downsampling
- Guidance scale: 5.0-7.5 recommended

Reference: Kwai, "Kolors: Effective Training of Diffusion Model for Photorealistic
Text-to-Image Synthesis", 2024

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `KolorsModel(NeuralNetworkArchitecture<>,DiffusionModelOptions<>,INoiseScheduler<>,UNetNoisePredictor<>,StandardVAE<>,IConditioningModule<>,Nullable<Int32>)` | Initializes a new instance of KolorsModel with full customization support. |

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

