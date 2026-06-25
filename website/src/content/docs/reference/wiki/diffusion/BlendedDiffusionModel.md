---
title: "BlendedDiffusionModel<T>"
description: "Blended Diffusion model for text-guided local image editing within user-specified masks."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Diffusion.ImageEditing`

Blended Diffusion model for text-guided local image editing within user-specified masks.

## For Beginners

Blended Diffusion lets you edit specific areas of an image with text.

How it works:

1. You provide an image, a mask marking the area to edit, and a text prompt
2. At each denoising step, new content is generated in the masked region
3. The generated content is blended with the original image in unmasked regions
4. The result is a seamless edit confined to the masked area

Key characteristics:

- Based on Stable Diffusion 1.5 (512x512, CLIP ViT-L/14)
- User-specified masks control where edits are applied
- Per-step blending ensures seamless transitions
- Uses DDPM scheduler for stochastic blending quality

## How It Works

Blended Diffusion enables text-guided local editing by combining CLIP-guided diffusion
with spatial blending. At each denoising step, the model generates content guided by the
text prompt in masked regions and blends it with the original image content in unmasked
regions, producing seamless localized edits.

Technical specifications:

- Architecture: SD 1.5 U-Net with per-step spatial blending
- Text encoder: CLIP ViT-L/14 (768-dim, 77 max tokens)
- Cross-attention dimension: 768
- VAE: 4 latent channels, scale factor 0.18215
- Noise schedule: Scaled linear beta [0.00085, 0.012], 1000 timesteps
- Scheduler: DDPM (stochastic for natural blending transitions)

Reference: Avrahami et al., "Blended Diffusion for Text-driven Editing of Natural Images", CVPR 2022

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `BlendedDiffusionModel(NeuralNetworkArchitecture<>,DiffusionModelOptions<>,INoiseScheduler<>,UNetNoisePredictor<>,StandardVAE<>,IConditioningModule<>,Nullable<Int32>)` | Initializes a new instance of BlendedDiffusionModel with full customization support. |

## Properties

| Property | Summary |
|:-----|:--------|
| `Conditioner` |  |
| `CrossAttentionDim` | Gets the cross-attention dimension (768 for CLIP ViT-L/14). |
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
| `InitializeLayers(UNetNoisePredictor<>,StandardVAE<>,Nullable<Int32>)` | Initializes the U-Net and VAE layers using custom layers if provided, or creating industry-standard SD 1.5 layers. |
| `SetParameters(Vector<>)` |  |

## Fields

| Field | Summary |
|:-----|:--------|
| `CROSS_ATTENTION_DIM` | Cross-attention dimension matching CLIP ViT-L/14 output (768). |
| `DEFAULT_GUIDANCE_SCALE` | Default classifier-free guidance scale (7.5). |
| `DefaultHeight` | Default image height for Blended Diffusion (SD 1.5 native resolution). |
| `DefaultWidth` | Default image width for Blended Diffusion (SD 1.5 native resolution). |
| `LATENT_CHANNELS` | Number of latent channels in the VAE. |

