---
title: "DallE2Model<T>"
description: "DALL-E 2 (unCLIP) model for text-to-image generation via CLIP-guided diffusion."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Diffusion.TextToImage`

DALL-E 2 (unCLIP) model for text-to-image generation via CLIP-guided diffusion.

## For Beginners

DALL-E 2 generates images through CLIP space:

How DALL-E 2 works:

1. Text is encoded by CLIP ViT-L/14 text encoder (768-dim)
2. A diffusion prior maps text embeddings → CLIP image embeddings
3. A diffusion decoder generates 64x64 images from image embeddings
4. Two upsampler stages scale to 256x256 → 1024x1024

Key characteristics:

- Two-stage: Diffusion prior + Diffusion decoder
- CLIP-guided: operates in CLIP embedding space
- Text encoder: CLIP ViT-L/14 (768-dim)
- Prior: Diffusion transformer mapping text→image embeddings
- Decoder: Modified GLIDE model, pixel-space diffusion
- Supports image variations (re-generate from CLIP embedding)

Advantages:

- Natural image variations through CLIP space manipulation
- Good compositional understanding
- Supports text-guided image editing

Limitations:

- Not open-source (proprietary to OpenAI)
- Superseded by DALL-E 3 in quality
- Sometimes struggles with text rendering

## How It Works

DALL-E 2 is a text-to-image model developed by OpenAI that uses a two-stage pipeline:
a prior that generates CLIP image embeddings from text, and a decoder that generates
images from those embeddings. This approach is also known as "unCLIP".

Technical specifications:

- Architecture: Diffusion Prior + Diffusion Decoder (unCLIP)
- Prior: Transformer-based diffusion, maps CLIP text→image embeddings
- Decoder: Modified GLIDE U-Net, ~3.5B parameters, 64x64 base
- Text encoder: CLIP ViT-L/14 (768-dim embeddings)
- Image encoder: CLIP ViT-L/14 (768-dim, used for prior training)
- Noise schedule: Linear beta, 1000 training timesteps
- Upsampler: Two ADM upsampler stages (64→256→1024)

Reference: Ramesh et al., "Hierarchical Text-Conditional Image Generation with CLIP Latents", 2022

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `DallE2Model(NeuralNetworkArchitecture<>,DiffusionModelOptions<>,INoiseScheduler<>,UNetNoisePredictor<>,UNetNoisePredictor<>,StandardVAE<>,IConditioningModule<>,Nullable<Int32>)` | Initializes a new instance of DallE2Model with full customization support. |

## Properties

| Property | Summary |
|:-----|:--------|
| `ClipDimension` | Gets the CLIP embedding dimension (768). |
| `Conditioner` |  |
| `DiffusionPrior` | Gets the diffusion prior that maps text embeddings to CLIP image embeddings. |
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
| `InitializeLayers(UNetNoisePredictor<>,UNetNoisePredictor<>,StandardVAE<>,Nullable<Int32>)` | Initializes the diffusion prior, decoder U-Net, and optional VAE, using custom layers from the user if provided or creating industry-standard layers from the unCLIP/DALL-E 2 paper. |
| `SetParameters(Vector<>)` |  |

## Fields

| Field | Summary |
|:-----|:--------|
| `DALLE2_CLIP_DIM` | CLIP embedding dimension (768 for ViT-L/14). |
| `DALLE2_DEFAULT_GUIDANCE_SCALE` | Default guidance scale for DALL-E 2 (4.0). |
| `DefaultHeight` | Default image height for DALL-E 2 base decoder (64x64). |
| `DefaultWidth` | Default image width for DALL-E 2 base decoder (64x64). |

