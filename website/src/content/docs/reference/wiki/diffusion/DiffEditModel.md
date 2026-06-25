---
title: "DiffEditModel<T>"
description: "DiffEdit model for automatic mask generation and text-guided image editing."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Diffusion.ImageEditing`

DiffEdit model for automatic mask generation and text-guided image editing.

## For Beginners

DiffEdit lets you edit images by just describing what to change.

How it works:

1. You provide the original image, a source prompt, and a target prompt
2. The model predicts noise for both prompts and computes the difference
3. The difference map becomes an automatic editing mask
4. The image is re-generated in masked regions guided by the target prompt

Key characteristics:

- Based on Stable Diffusion 1.5 (512x512, CLIP ViT-L/14)
- No manual mask required -- masks are generated automatically
- Preserves unedited regions faithfully
- Uses DDIM scheduler for deterministic inversion and editing

## How It Works

DiffEdit automatically generates editing masks by comparing noise predictions conditioned
on source and target text prompts. The difference in predictions highlights which regions
need to change, producing a spatial mask without any manual annotation. The mask is then
used to selectively apply the edit while preserving unrelated regions.

Technical specifications:

- Architecture: SD 1.5 U-Net with differential noise prediction for mask generation
- Text encoder: CLIP ViT-L/14 (768-dim, 77 max tokens)
- Cross-attention dimension: 768
- VAE: 4 latent channels, scale factor 0.18215
- Noise schedule: Scaled linear beta [0.00085, 0.012], 1000 timesteps
- Scheduler: DDIM (deterministic inversion required)

Reference: Couairon et al., "DiffEdit: Diffusion-based Semantic Image Editing with Mask Guidance", ICLR 2023

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `DiffEditModel(NeuralNetworkArchitecture<>,DiffusionModelOptions<>,INoiseScheduler<>,UNetNoisePredictor<>,StandardVAE<>,IConditioningModule<>,Nullable<Int32>)` | Initializes a new instance of DiffEditModel with full customization support. |

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
| `DefaultHeight` | Default image height for DiffEdit (SD 1.5 native resolution). |
| `DefaultWidth` | Default image width for DiffEdit (SD 1.5 native resolution). |
| `LATENT_CHANNELS` | Number of latent channels in the VAE. |

