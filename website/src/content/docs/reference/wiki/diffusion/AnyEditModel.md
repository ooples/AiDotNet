---
title: "AnyEditModel<T>"
description: "AnyEdit model for handling diverse editing types through unified instruction understanding."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Diffusion.ImageEditing`

AnyEdit model for handling diverse editing types through unified instruction understanding.

## For Beginners

AnyEdit is versatile — it can handle almost any type of image
edit you describe in words. Change colors, add objects, modify styles, or adjust
specific regions, all with the same model and natural language instructions.

## How It Works

AnyEdit supports a wide range of editing operations including local/global edits,
style changes, action edits, and visual concept manipulation through a unified model
trained on diverse editing instruction-image pairs. Based on InstructPix2Pix with SD1.5.

Technical specifications:

- Base model: Stable Diffusion 1.5 (InstructPix2Pix variant)
- Text encoder: CLIP ViT-L/14 (768-dim)
- Input channels: 8 (4 latent + 4 source image latent)
- Edit types: local, global, style, action, visual concept
- Training: 2.5M diverse instruction-image pairs

Reference: Yu et al., "AnyEdit: Mastering Unified High-Quality Image Editing for Any Idea", 2024

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

