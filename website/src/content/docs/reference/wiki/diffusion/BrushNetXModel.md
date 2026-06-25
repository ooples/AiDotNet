---
title: "BrushNetXModel<T>"
description: "BrushNet-X model extending BrushNet to SDXL architecture for high-resolution inpainting."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Diffusion.ImageEditing`

BrushNet-X model extending BrushNet to SDXL architecture for high-resolution inpainting.

## For Beginners

This is a higher-resolution version of BrushNet built on SDXL.
It generates larger, more detailed inpainting results at 1024x1024 resolution compared
to BrushNet's 512x512, with better text understanding from dual encoders.

## How It Works

BrushNet-X adapts the dual-branch BrushNet inpainting approach to the SDXL architecture,
enabling high-resolution (1024x1024) inpainting with improved coherence and detail.
Uses SDXL's dual text encoders (CLIP-L + CLIP-G) for enhanced prompt understanding.

Technical specifications:

- Architecture: Dual-branch U-Net (SDXL backbone)
- Text encoders: CLIP ViT-L/14 (768-dim) + OpenCLIP ViT-G/14 (1280-dim) = 2048 context
- Input channels: 9 (4 latent + 4 masked image latent + 1 mask)
- Base channels: 320, multipliers [1, 2, 4]
- Resolution: 1024x1024 native

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

