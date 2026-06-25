---
title: "LEDITSPPModel<T>"
description: "LEDITS++ model for precise multi-concept editing of real images with automatic masking."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Diffusion.ImageEditing`

LEDITS++ model for precise multi-concept editing of real images with automatic masking.

## For Beginners

LEDITS++ lets you edit multiple things in a photo at once.

How it works:

1. The real image is inverted into latent space using DDPM-based inversion
2. Multiple editing prompts can be applied simultaneously
3. Automatic masking determines which regions each edit affects
4. Edits are blended using semantic guidance for natural results

Key characteristics:

- Based on Stable Diffusion 1.5 (512x512, CLIP ViT-L/14)
- Supports multiple simultaneous edits (e.g., change hair AND add glasses)
- Automatic semantic masking per concept
- Uses DPM-Solver++ for fast, high-quality sampling (15-25 steps)

## How It Works

LEDITS++ (Lightweight and Efficient Diffusion Inter-Image Transformations with Semantic guidance++)
enables simultaneous editing of multiple concepts in real images. It combines DDPM inversion
with semantic grounding to automatically determine which image regions correspond to each
editing concept, enabling precise multi-edit operations without manual masks.

Technical specifications:

- Architecture: SD 1.5 U-Net with multi-concept semantic guidance
- Text encoder: CLIP ViT-L/14 (768-dim, 77 max tokens)
- Cross-attention dimension: 768
- VAE: 4 latent channels, scale factor 0.18215
- Noise schedule: Scaled linear beta [0.00085, 0.012], 1000 timesteps
- Scheduler: DPM-Solver++ multistep (fast convergence)

Reference: Brack et al., "LEDITS++: Limitless Image Editing using Text-to-Image Models", CVPR 2024

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `LEDITSPPModel(NeuralNetworkArchitecture<>,DiffusionModelOptions<>,INoiseScheduler<>,UNetNoisePredictor<>,StandardVAE<>,IConditioningModule<>,Nullable<Int32>)` | Initializes a new instance of LEDITSPPModel with full customization support. |

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
| `DefaultHeight` | Default image height for LEDITS++ (SD 1.5 native resolution). |
| `DefaultWidth` | Default image width for LEDITS++ (SD 1.5 native resolution). |
| `LATENT_CHANNELS` | Number of latent channels in the VAE. |

