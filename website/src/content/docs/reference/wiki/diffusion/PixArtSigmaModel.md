---
title: "PixArtSigmaModel<T>"
description: "PixArt-Sigma model for high-resolution text-to-image generation with improved quality."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Diffusion.TextToImage`

PixArt-Sigma model for high-resolution text-to-image generation with improved quality.

## For Beginners

PixArt-Sigma is an upgraded version of PixArt-Alpha:

Key improvements over PixArt-Alpha:

- Higher native resolution: supports up to 4096x4096
- Better image quality from improved training data
- Flexible aspect ratios via bucket training
- Still very efficient (DiT-based, much faster than U-Net models)

How PixArt-Sigma works:

1. Text goes through T5-XXL encoder (4096-dim)
2. DiT transformer blocks denoise the latent with cross-attention to text
3. Trained on high-quality curated datasets with better captions
4. VAE decodes to final high-resolution image

Use PixArt-Sigma when you need:

- High-resolution output (up to 4K)
- Fast generation on limited hardware
- Good quality with flexible aspect ratios

## How It Works

PixArt-Sigma is the successor to PixArt-Alpha, featuring improved training data quality,
higher native resolution support (up to 4K), and better aesthetic quality. It maintains
the efficient DiT architecture while achieving quality comparable to SDXL and DALL-E 3.

Technical specifications:

- Architecture: DiT (Diffusion Transformer) with T5-XXL text encoder
- Parameters: ~600M (DiT-XL/2)
- Text encoder: T5-XXL (4096-dim embeddings)
- Native resolution: up to 4096x4096
- Latent space: 4 channels, 8x downsampling
- Scheduler: DPM-Solver with 20 steps recommended

Reference: Chen et al., "PixArt-Sigma: Weak-to-Strong Training of Diffusion Transformer
for 4K Text-to-Image Generation", ECCV 2024

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `PixArtSigmaModel(NeuralNetworkArchitecture<>,DiffusionModelOptions<>,INoiseScheduler<>,DiTNoisePredictor<>,StandardVAE<>,IConditioningModule<>,Nullable<Int32>)` | Initializes a new instance of PixArtSigmaModel with full customization support. |

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

