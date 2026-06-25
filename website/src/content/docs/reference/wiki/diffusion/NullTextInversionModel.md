---
title: "NullTextInversionModel<T>"
description: "Null-text Inversion model for editing real images by optimizing unconditional embeddings."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Diffusion.ImageEditing`

Null-text Inversion model for editing real images by optimizing unconditional embeddings.

## For Beginners

Null-text Inversion lets you edit real photos with text prompts.

How it works:

1. DDIM inversion maps your real image back to noise (latent code)
2. The null-text embedding is optimized per-timestep for accurate reconstruction
3. Editing is performed by swapping the text prompt while keeping the optimized embeddings

Key characteristics:

- Based on Stable Diffusion 1.5 (512x512, CLIP ViT-L/14)
- Enables Prompt-to-Prompt editing on real images
- Requires per-image optimization (~50 optimization steps per timestep)
- Uses DDIM scheduler for deterministic inversion and generation

## How It Works

Null-text Inversion enables precise editing of real photographs by first inverting them
into latent space using DDIM inversion, then optimizing the unconditional (null-text)
embedding at each timestep to ensure faithful reconstruction. Once inverted, standard
text-guided editing techniques like Prompt-to-Prompt can modify the image.

Technical specifications:

- Architecture: SD 1.5 U-Net with per-timestep null-text optimization
- Text encoder: CLIP ViT-L/14 (768-dim, 77 max tokens)
- Cross-attention dimension: 768
- VAE: 4 latent channels, scale factor 0.18215
- Noise schedule: Scaled linear beta [0.00085, 0.012], 1000 timesteps
- Scheduler: DDIM (required for deterministic inversion)

Reference: Mokady et al., "Null-text Inversion for Editing Real Images using Guided Diffusion Models", CVPR 2023

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `NullTextInversionModel(NeuralNetworkArchitecture<>,DiffusionModelOptions<>,INoiseScheduler<>,UNetNoisePredictor<>,StandardVAE<>,IConditioningModule<>,Nullable<Int32>)` | Initializes a new instance of NullTextInversionModel with full customization support. |

## Properties

| Property | Summary |
|:-----|:--------|
| `Conditioner` |  |
| `CrossAttentionDim` | Gets the cross-attention dimension (768 for CLIP ViT-L/14). |
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
| `InitializeLayers(UNetNoisePredictor<>,StandardVAE<>,Nullable<Int32>)` | Initializes the U-Net and VAE layers using custom layers if provided, or creating industry-standard SD 1.5 layers. |
| `SetParameters(Vector<>)` |  |

## Fields

| Field | Summary |
|:-----|:--------|
| `CROSS_ATTENTION_DIM` | Cross-attention dimension matching CLIP ViT-L/14 output (768). |
| `DEFAULT_GUIDANCE_SCALE` | Default classifier-free guidance scale (7.5). |
| `DefaultHeight` | Default image height for Null-text Inversion (SD 1.5 native resolution). |
| `DefaultWidth` | Default image width for Null-text Inversion (SD 1.5 native resolution). |
| `LATENT_CHANNELS` | Number of latent channels in the VAE. |

