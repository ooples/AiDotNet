---
title: "Imagen2Model<T>"
description: "Imagen 2 model for improved cascaded text-to-image generation."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Diffusion.TextToImage`

Imagen 2 model for improved cascaded text-to-image generation.

## For Beginners

Imagen 2 is Google's upgraded image generator:

Key improvements over Imagen 1:

- Much better text rendering in images
- Improved photorealism and detail
- Better safety filtering
- Enhanced prompt following

How Imagen 2 works:

1. Text encoder produces rich embeddings
2. Base model generates low-resolution image (64x64)
3. Cascaded super-resolution: 64 -> 256 -> 1024
4. Each stage uses U-Net with cross-attention to text

Use Imagen 2 when you need:

- Photorealistic image generation
- Text rendered correctly in images
- High fidelity to complex prompts

## How It Works

Imagen 2 is Google DeepMind's improved text-to-image model, successor to the original Imagen.
It features improved image quality, better text rendering, and enhanced prompt following.

Technical specifications:

- Architecture: Cascaded latent diffusion with Efficient U-Net
- Base model: 64x64 generation
- Super-resolution stages: 64->256, 256->1024
- Text encoder: T5-XXL (frozen, 4096-dim)
- Dynamic thresholding for high guidance scales

Reference: Google DeepMind, "Imagen 2", 2023

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `Imagen2Model(NeuralNetworkArchitecture<>,DiffusionModelOptions<>,INoiseScheduler<>,UNetNoisePredictor<>,StandardVAE<>,IConditioningModule<>,Boolean,Nullable<Int32>)` | Initializes a new instance of Imagen2Model with full customization support. |

## Properties

| Property | Summary |
|:-----|:--------|
| `Conditioner` |  |
| `IsImagen3` | Gets whether this is the Imagen 3 variant. |
| `LatentChannels` |  |
| `NoisePredictor` |  |
| `ParameterCount` | Counts the flat-API parameter surface (predictor + VAE). |
| `VAE` |  |

## Methods

| Method | Summary |
|:-----|:--------|
| `Clone` |  |
| `CreateImagen3(Nullable<Int32>)` | Creates an Imagen 3 model instance. |
| `DeepCopy` |  |
| `GenerateFromText(String,String,Int32,Int32,Int32,Nullable<Double>,Nullable<Int32>)` |  |
| `GetModelMetadata` |  |
| `GetParameters` |  |
| `SetParameters(Vector<>)` |  |

