---
title: "PhotoMakerModel<T>"
description: "PhotoMaker model — identity-customized photo generation with stacked ID embedding."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Diffusion.Control`

PhotoMaker model — identity-customized photo generation with stacked ID embedding.

## For Beginners

PhotoMaker creates personalized photos from a few reference images:

Key characteristics:

- 1-4 reference images for identity (no fine-tuning needed)
- Stacked ID embedding: fuses multiple reference features
- CLIP image encoder for identity extraction
- Works with SDXL for high-quality output

Use PhotoMaker when you need:

- Customized photo generation from few references
- Identity-consistent character images
- Quick personalization without training

## How It Works

PhotoMaker generates customized photos of a person using 1-4 reference images.
It uses a stacked ID embedding approach that fuses identity features from multiple
reference images into the text conditioning pipeline.

Technical specifications:

- Architecture: SDXL + stacked ID embedding
- Identity encoder: CLIP ViT-L/14 (fine-tuned)
- Base model: SDXL U-Net
- Resolution: 1024x1024

Reference: Li et al., "PhotoMaker: Customizing Realistic Human Photos
via Stacked ID Embedding", CVPR 2024

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `PhotoMakerModel(NeuralNetworkArchitecture<>,DiffusionModelOptions<>,INoiseScheduler<>,UNetNoisePredictor<>,StandardVAE<>,IConditioningModule<>,Nullable<Int32>)` | Initializes a new instance of the `PhotoMakerModel` class. |

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
| `InitializeLayers(UNetNoisePredictor<>,StandardVAE<>,Nullable<Int32>)` | Initializes the U-Net noise predictor and VAE components. |
| `SetParameters(Vector<>)` |  |

## Fields

| Field | Summary |
|:-----|:--------|
| `CROSS_ATTENTION_DIM` | Cross-attention dimension for SDXL (2048). |
| `DEFAULT_GUIDANCE_SCALE` | Default guidance scale for identity-preserving generation. |
| `DefaultHeight` | Default height for PhotoMaker generation (SDXL native). |
| `DefaultWidth` | Default width for PhotoMaker generation (SDXL native). |
| `LATENT_CHANNELS` | Number of latent channels in the VAE latent space. |

