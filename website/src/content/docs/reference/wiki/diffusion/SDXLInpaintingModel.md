---
title: "SDXLInpaintingModel<T>"
description: "SDXL Inpainting model for high-resolution 1024x1024 mask-based image inpainting."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Diffusion.ImageEditing`

SDXL Inpainting model for high-resolution 1024x1024 mask-based image inpainting.

## For Beginners

This is Stable Diffusion XL's built-in inpainting model. You
provide an image and paint a mask over what you want to change, then describe what
should appear there. It generates results at 1024x1024 resolution with excellent
text understanding from dual CLIP encoders.

## How It Works

The official SDXL inpainting model uses a 9-channel input (4 latent + 4 masked image + 1 mask)
with the full SDXL U-Net architecture. It provides native 1024x1024 inpainting with the
quality and prompt following of SDXL's dual text encoder conditioning.

Technical specifications:

- Architecture: SDXL U-Net with 9-channel input
- Text encoders: CLIP ViT-L/14 (768-dim) + OpenCLIP ViT-G/14 (1280-dim) = 2048 context
- Input channels: 9 (4 latent + 4 masked image latent + 1 binary mask)
- Base channels: 320, multipliers [1, 2, 4]
- Resolution: 1024x1024 native
- Scheduler: Euler Discrete (SDXL default)

Reference: Podell et al., "SDXL: Improving Latent Diffusion Models for High-Resolution Image Synthesis", ICLR 2024

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
| `GetModelMetadata` |  |
| `GetParameters` |  |
| `SetParameters(Vector<>)` |  |

