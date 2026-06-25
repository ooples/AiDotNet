---
title: "SDEditModel<T>"
description: "SDEdit model for image-guided synthesis via partial noise injection and guided denoising."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Diffusion.ImageEditing`

SDEdit model for image-guided synthesis via partial noise injection and guided denoising.

## For Beginners

SDEdit turns rough sketches and images into polished results.

How it works:

1. You provide an input image (sketch, rough edit, or photograph)
2. The model adds noise to partially corrupt the image
3. The diffusion model denoises it, guided by text prompts
4. The result balances input structure with photorealistic output

Key characteristics:

- Based on Stable Diffusion 1.5 (512x512, CLIP ViT-L/14)
- No fine-tuning needed -- works with any pre-trained diffusion model
- Noise strength controls realism vs. faithfulness tradeoff
- Uses DDPM scheduler for stochastic denoising

## How It Works

SDEdit (Stochastic Differential Editing) transforms sketches, strokes, and existing images
into realistic outputs by adding controlled amounts of noise to the input and then denoising
with a diffusion model. The noise strength controls the balance between faithfulness to the
input guide and realism of the output.

Technical specifications:

- Architecture: SD 1.5 U-Net with partial noise injection pipeline
- Text encoder: CLIP ViT-L/14 (768-dim, 77 max tokens)
- Cross-attention dimension: 768
- VAE: 4 latent channels, scale factor 0.18215
- Noise schedule: Scaled linear beta [0.00085, 0.012], 1000 timesteps
- Scheduler: DDPM (stochastic denoising for natural output variation)

Reference: Meng et al., "SDEdit: Guided Image Synthesis and Editing with Stochastic Differential Equations", ICLR 2022

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `SDEditModel(NeuralNetworkArchitecture<>,DiffusionModelOptions<>,INoiseScheduler<>,UNetNoisePredictor<>,StandardVAE<>,IConditioningModule<>,Nullable<Int32>)` | Initializes a new instance of SDEditModel with full customization support. |

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
| `DefaultHeight` | Default image height for SDEdit (SD 1.5 native resolution). |
| `DefaultWidth` | Default image width for SDEdit (SD 1.5 native resolution). |
| `LATENT_CHANNELS` | Number of latent channels in the VAE. |

