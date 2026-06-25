---
title: "PaintByExampleModel<T>"
description: "Paint-by-Example model for exemplar-based inpainting using reference images."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Diffusion.ImageEditing`

Paint-by-Example model for exemplar-based inpainting using reference images.

## For Beginners

Paint-by-Example fills in parts of an image using another image as a guide.

How it works:

1. You provide a source image, a mask indicating the region to fill, and a reference image
2. The reference image is encoded by CLIP into a visual embedding
3. The U-Net takes 9 input channels (latent + masked image + mask) and denoises
4. The filled region matches the style and content of the reference image

Key characteristics:

- Based on Stable Diffusion 1.5 (512x512)
- U-Net has 9 input channels (4 latent + 4 masked image + 1 mask)
- Uses exemplar images instead of text for conditioning
- Uses DDIM scheduler for deterministic inpainting results

## How It Works

Paint-by-Example fills masked image regions using exemplar images as visual references
instead of text prompts. The U-Net receives 9 input channels: 4 latent channels from the
noisy latent, 4 channels from the masked source image latent, and 1 channel for the binary
mask. The exemplar image is encoded via CLIP and injected through cross-attention.

Technical specifications:

- Architecture: SD 1.5 U-Net modified with 9 input channels for inpainting
- Conditioning: CLIP image encoder (768-dim) via cross-attention
- Input channels: 9 (4 noisy latent + 4 masked source latent + 1 binary mask)
- Output channels: 4 (latent space prediction)
- Cross-attention dimension: 768
- VAE: 4 latent channels, scale factor 0.18215
- Noise schedule: Scaled linear beta [0.00085, 0.012], 1000 timesteps
- Scheduler: DDIM

Reference: Yang et al., "Paint by Example: Exemplar-based Image Editing with Diffusion Models", CVPR 2023

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `PaintByExampleModel(NeuralNetworkArchitecture<>,DiffusionModelOptions<>,INoiseScheduler<>,UNetNoisePredictor<>,StandardVAE<>,IConditioningModule<>,Nullable<Int32>)` | Initializes a new instance of PaintByExampleModel with full customization support. |

## Properties

| Property | Summary |
|:-----|:--------|
| `Conditioner` |  |
| `CrossAttentionDim` | Gets the cross-attention dimension (768 for CLIP image encoder). |
| `InputChannels` | Gets the number of U-Net input channels (9: 4 latent + 4 masked + 1 mask). |
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
| `InitializeLayers(UNetNoisePredictor<>,StandardVAE<>,Nullable<Int32>)` | Initializes the U-Net and VAE layers using custom layers if provided, or creating industry-standard layers for exemplar-based inpainting. |
| `SetParameters(Vector<>)` |  |

## Fields

| Field | Summary |
|:-----|:--------|
| `CROSS_ATTENTION_DIM` | Cross-attention dimension matching CLIP image encoder output (768). |
| `DEFAULT_GUIDANCE_SCALE` | Default classifier-free guidance scale (7.5). |
| `DefaultHeight` | Default image height for Paint-by-Example (SD 1.5 native resolution). |
| `DefaultWidth` | Default image width for Paint-by-Example (SD 1.5 native resolution). |
| `INPUT_CHANNELS` | Number of U-Net input channels (4 latent + 4 masked image + 1 mask). |
| `LATENT_CHANNELS` | Number of latent channels in the VAE. |

