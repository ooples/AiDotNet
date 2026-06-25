---
title: "ReplaceAnythingModel<T>"
description: "ReplaceAnything model for text-guided object replacement within SAM-segmented regions."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Diffusion.ImageEditing`

ReplaceAnything model for text-guided object replacement within SAM-segmented regions.

## For Beginners

Click on an object in your image (a car, a tree, a person),
describe what you want instead ("a red sports car", "a cherry blossom tree"), and
ReplaceAnything seamlessly swaps it out while keeping the rest of the image intact.

## How It Works

Combines SAM-based segmentation with SD1.5 inpainting diffusion to automatically replace
any segmented object with text-described content. The user selects an object (via
point, box, or mask) and provides a text description of the replacement.

Technical specifications:

- Pipeline: SAM segmentation -> mask generation -> SD1.5 inpainting
- Segmentation: Segment Anything Model (SAM) for interactive selection
- Inpainting: SD1.5 with 9-channel input (4 latent + 4 masked + 1 mask)
- Text encoder: CLIP ViT-L/14 (768-dim)
- Selection modes: point, bounding box, or manual mask

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

