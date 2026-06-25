---
title: "FreeInpaintModel<T>"
description: "Free-form inpainting model using masked diffusion with irregular mask support."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Diffusion.ImageEditing`

Free-form inpainting model using masked diffusion with irregular mask support.

## For Beginners

Most inpainting works with simple rectangular masks. Free
inpainting lets you draw any shape — circles, squiggles, complex outlines — and
fills them in naturally. Great for detailed retouching and creative editing.

## How It Works

Supports arbitrary free-form masks of any shape, not just rectangular regions.
Uses mask-aware attention in the SD1.5 U-Net to ensure generated content respects
irregular boundaries while maintaining visual coherence with surrounding context.

Technical specifications:

- Base model: Stable Diffusion 1.5 inpainting
- Text encoder: CLIP ViT-L/14 (768-dim)
- Input channels: 9 (4 latent + 4 masked image latent + 1 mask)
- Mask support: Arbitrary free-form shapes
- Mask-aware attention for boundary coherence

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

