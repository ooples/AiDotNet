---
title: "CogView4Model<T>"
description: "CogView-4 model for bilingual text-to-image generation by Zhipu AI."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Diffusion.TextToImage`

CogView-4 model for bilingual text-to-image generation by Zhipu AI.

## For Beginners

CogView-4 is a model from Zhipu AI (Tsinghua University spinoff) that
understands both Chinese and English prompts equally well.

How CogView-4 works:

1. Text is encoded by GLM (General Language Model) with bilingual support
2. A SiT (Scalable Interpolant Transformer) processes text and noise tokens
3. Relay diffusion enables efficient high-resolution generation in two stages
4. A 16-channel VAE decodes latents to full images

Key characteristics:

- SiT architecture with scalable interpolant formulation
- GLM text encoder for bilingual Chinese-English understanding
- Relay diffusion: low-res generation → high-res refinement
- 16 latent channels
- Strong understanding of Asian cultural contexts
- 30 inference steps typical

Advantages:

- Best-in-class bilingual prompt understanding
- Strong cultural context for Asian aesthetics
- Relay diffusion enables efficient high-res generation
- Open-weight model

Limitations:

- Smaller English-only community
- Less LoRA/adapter ecosystem
- Relay diffusion adds complexity

## How It Works

CogView-4 uses a Scalable Interpolant Transformer (SiT) architecture with relay diffusion
for high-resolution image generation. It features bilingual Chinese-English text understanding
through a GLM-based text encoder, providing strong comprehension of both languages and
Asian cultural contexts.

Technical specifications:

- Architecture: SiT with relay diffusion
- Text encoder: GLM (bilingual, 4096-dim)
- Hidden size: 2048, 24 layers, 32 heads
- Latent channels: 16
- Training: Continuous-time diffusion with interpolant formulation
- Default: 30 steps, guidance scale 7.5
- Resolution: 1024x1024 default

Reference: Zheng et al., "CogView-4: Bilingual Text-to-Image Generation
with Relay Diffusion", 2024

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `CogView4Model(NeuralNetworkArchitecture<>,DiffusionModelOptions<>,INoiseScheduler<>,SiTPredictor<>,StandardVAE<>,IConditioningModule<>,Nullable<Int32>)` | Initializes a new instance of CogView4Model. |

## Properties

| Property | Summary |
|:-----|:--------|
| `Conditioner` |  |
| `LatentChannels` |  |
| `NoisePredictor` |  |
| `ParameterCount` |  |
| `SupportsBilingual` | Gets whether this model supports bilingual Chinese-English prompts. |
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
| `DefaultHeight` | Default image height for CogView-4 (1024x1024). |
| `DefaultWidth` | Default image width for CogView-4 (1024x1024). |

