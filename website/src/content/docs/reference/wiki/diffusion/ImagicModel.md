---
title: "ImagicModel<T>"
description: "Imagic model for text-aligned real image editing via embedding optimization and model fine-tuning."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Diffusion.ImageEditing`

Imagic model for text-aligned real image editing via embedding optimization and model fine-tuning.

## For Beginners

Imagic edits real photos to match your text description.

How it works:

1. A text embedding is optimized to faithfully reconstruct the original image
2. The diffusion model is fine-tuned to bind the embedding to the image content
3. Interpolation between optimized and target embeddings applies the desired edit

Key characteristics:

- Based on Stable Diffusion 1.5 (512x512, CLIP ViT-L/14)
- Can perform complex edits (pose changes, object addition, style transfer)
- Requires per-image optimization and fine-tuning
- Uses DDIM scheduler for deterministic generation

## How It Works

Imagic enables sophisticated editing of real images to match a target text description through
a three-stage process: (1) optimizing a text embedding to reconstruct the input image,
(2) fine-tuning the diffusion model with the optimized embedding, and (3) interpolating
between the optimized and target embeddings to produce the desired edit.

Technical specifications:

- Architecture: SD 1.5 U-Net with text embedding optimization + model fine-tuning
- Text encoder: CLIP ViT-L/14 (768-dim, 77 max tokens)
- Cross-attention dimension: 768
- VAE: 4 latent channels, scale factor 0.18215
- Noise schedule: Scaled linear beta [0.00085, 0.012], 1000 timesteps
- Scheduler: DDIM (deterministic for consistent interpolation results)

Reference: Kawar et al., "Imagic: Text-Based Real Image Editing with Diffusion Models", CVPR 2023

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `ImagicModel(NeuralNetworkArchitecture<>,DiffusionModelOptions<>,INoiseScheduler<>,UNetNoisePredictor<>,StandardVAE<>,IConditioningModule<>,Nullable<Int32>)` | Initializes a new instance of ImagicModel with full customization support. |

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
| `DefaultHeight` | Default image height for Imagic (SD 1.5 native resolution). |
| `DefaultWidth` | Default image width for Imagic (SD 1.5 native resolution). |
| `LATENT_CHANNELS` | Number of latent channels in the VAE. |

