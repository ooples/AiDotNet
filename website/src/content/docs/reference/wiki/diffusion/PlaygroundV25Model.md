---
title: "PlaygroundV25Model<T>"
description: "Playground v2.5 model for aesthetically-focused text-to-image generation."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Diffusion.TextToImage`

Playground v2.5 model for aesthetically-focused text-to-image generation.

## For Beginners

Playground v2.5 focuses on making the most visually appealing images:

How Playground v2.5 works:

1. Uses the same SDXL architecture (dual text encoders, U-Net)
2. Trained with enhanced aesthetic filtering and human preference alignment
3. Uses EDM (Elucidated Diffusion Models) training framework
4. Generates high-quality 1024x1024 images with exceptional aesthetics

Key characteristics:

- SDXL-compatible architecture (drop-in replacement)
- Dual text encoders: CLIP ViT-L/14 (768-dim) + OpenCLIP ViT-bigG/14 (1280-dim)
- Combined cross-attention dimension: 2048
- Aesthetic-focused training with human preference data
- EDM (Elucidated Diffusion Models) noise schedule
- Native 1024x1024 resolution

Advantages:

- Superior aesthetic quality (highest scores on various benchmarks)
- SDXL-compatible: works with SDXL LoRAs and tools
- Open-source (Apache 2.0 license)
- Excellent for photorealistic and artistic images

Limitations:

- Slightly slower than standard SDXL due to larger training
- May over-aestheticize some outputs
- Fewer community fine-tunes than standard SDXL

## How It Works

Playground v2.5 is an open-source text-to-image model optimized for aesthetic quality,
developed by Playground AI. It achieves state-of-the-art aesthetic scores while maintaining
the SDXL architecture for broad compatibility.

Technical specifications:

- Architecture: SDXL-compatible U-Net with dual text encoders
- U-Net: ~2.6B parameters, base channels 320, multipliers [1, 2, 4]
- Text encoder 1: CLIP ViT-L/14 (768-dim)
- Text encoder 2: OpenCLIP ViT-bigG/14 (1280-dim)
- Combined context: 2048-dim (768 + 1280)
- VAE: SDXL VAE, 4 latent channels, scale factor 0.13025
- Training: EDM framework with aesthetic optimization
- Resolution: 1024x1024

Reference: Li et al., "Playground v2.5: Three Insights towards Enhancing Aesthetic Quality in Text-to-Image Generation", 2024

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `PlaygroundV25Model(NeuralNetworkArchitecture<>,DiffusionModelOptions<>,INoiseScheduler<>,UNetNoisePredictor<>,StandardVAE<>,IConditioningModule<>,Nullable<Int32>)` | Initializes a new instance of PlaygroundV25Model with full customization support. |

## Properties

| Property | Summary |
|:-----|:--------|
| `Conditioner` |  |
| `CrossAttentionDim` | Gets the cross-attention dimension (2048 for dual text encoders). |
| `LatentChannels` |  |
| `NoisePredictor` |  |
| `ParameterCount` | Counts the flat-API parameter surface (predictor + VAE). |
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
| `InitializeLayers(UNetNoisePredictor<>,StandardVAE<>,Nullable<Int32>)` | Initializes the U-Net and VAE layers following SDXL architecture, using custom layers from the user if provided or creating industry-standard layers from the Playground v2.5 paper. |
| `SetParameters(Vector<>)` |  |

## Fields

| Field | Summary |
|:-----|:--------|
| `DefaultHeight` | Default image height for Playground v2.5 (1024x1024). |
| `DefaultWidth` | Default image width for Playground v2.5 (1024x1024). |
| `PG_CROSS_ATTENTION_DIM` | Combined cross-attention dimension (CLIP 768 + OpenCLIP 1280 = 2048). |
| `PG_DEFAULT_GUIDANCE_SCALE` | Default guidance scale for Playground v2.5 (3.0, lower than SDXL). |

