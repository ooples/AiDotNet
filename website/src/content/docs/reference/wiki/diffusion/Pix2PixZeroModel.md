---
title: "Pix2PixZeroModel<T>"
description: "Pix2Pix-Zero model for zero-shot image-to-image translation without paired training data."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Diffusion.ImageEditing`

Pix2Pix-Zero model for zero-shot image-to-image translation without paired training data.

## For Beginners

Pix2Pix-Zero can translate images between domains (like
"cat to dog" or "day to night") without needing matched training pairs. It figures
out the transformation direction automatically from the text descriptions alone.

## How It Works

Pix2Pix-Zero performs image translation by discovering editing directions in the
text embedding space using sentence similarity, then applying these directions
during DDIM inversion-based editing. No paired training examples needed.
Uses standard SD1.5 backbone with cross-attention guidance.

Technical specifications:

- Base model: Stable Diffusion 1.5
- Text encoder: CLIP ViT-L/14 (768-dim)
- Input channels: 4 (standard latent)
- Editing: DDIM inversion + text embedding direction discovery
- Cross-attention guidance for structural preservation

Reference: Parmar et al., "Zero-shot Image-to-Image Translation", SIGGRAPH 2023

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

