---
title: "BrushEditModel<T>"
description: "BrushEdit model combining multimodal LLM understanding with BrushNet inpainting."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Diffusion.ImageEditing`

BrushEdit model combining multimodal LLM understanding with BrushNet inpainting.

## For Beginners

BrushEdit combines AI understanding with inpainting. Instead
of drawing masks yourself, you just describe what to change ("remove the car on the
left"), and BrushEdit figures out where to mask and what to generate.

## How It Works

BrushEdit uses a multimodal LLM to understand editing instructions and automatically
generate appropriate masks, then applies BrushNet-style dual-branch inpainting on SD1.5.
This enables natural language editing without manual mask creation.

Technical specifications:

- Pipeline: Multimodal LLM (mask generation) + BrushNet (dual-branch inpainting)
- Inpainting backbone: SD1.5 with BrushNet dual-branch architecture
- Text encoder: CLIP ViT-L/14 (768-dim)
- Input channels: 9 (4 latent + 4 masked image latent + 1 mask)
- Mask generation: Automatic from natural language via LLM

Reference: Li et al., "BrushEdit: All-In-One Image Inpainting and Editing", 2024

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

