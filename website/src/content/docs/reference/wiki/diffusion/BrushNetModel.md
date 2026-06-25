---
title: "BrushNetModel<T>"
description: "BrushNet model for plug-and-play dual-branch inpainting with any diffusion backbone."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Diffusion.ImageEditing`

BrushNet model for plug-and-play dual-branch inpainting with any diffusion backbone.

## For Beginners

BrushNet is a powerful inpainting model that works as an add-on to
existing diffusion models. It fills in masked regions of an image while keeping the unmasked
parts perfectly intact. The "dual-branch" design means it processes the context and the
generation separately, then combines them at every step.

## How It Works

BrushNet introduces a dual-branch architecture where one branch processes the masked image
and injects features into the main denoising branch at each layer. Unlike standard inpainting
that concatenates mask channels, BrushNet provides dense per-layer conditioning for superior
coherence at mask boundaries.

Technical specifications:

- Architecture: Dual-branch U-Net (SD1.5 backbone)
- Text encoder: CLIP ViT-L/14 (768-dim)
- Input channels: 9 (4 latent + 4 masked image latent + 1 mask)
- Base channels: 320, multipliers [1, 2, 4, 4]
- Auxiliary branch: Mirrors main U-Net encoder, injects at every layer
- Resolution: 512x512 native

Reference: Ju et al., "BrushNet: A Plug-and-Play Image Inpainting Model with Decomposed Dual-Branch Diffusion", 2024

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

