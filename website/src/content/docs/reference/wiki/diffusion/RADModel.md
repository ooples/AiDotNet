---
title: "RADModel<T>"
description: "Region-Aware Diffusion (RAD) model for spatially controlled inpainting and editing."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Diffusion.ImageEditing`

Region-Aware Diffusion (RAD) model for spatially controlled inpainting and editing.

## For Beginners

RAD lets you edit different parts of an image with different
instructions simultaneously. For example, you could change the sky to sunset while
changing the ground to snow, all in one generation pass.

## How It Works

RAD enables region-specific control during diffusion by applying different text prompts
to different spatial regions. Each region can have its own prompt and guidance scale,
allowing fine-grained multi-region editing in a single pass using SD1.5 backbone.

Technical specifications:

- Base model: Stable Diffusion 1.5 inpainting
- Text encoder: CLIP ViT-L/14 (768-dim)
- Input channels: 9 (4 latent + 4 masked image latent + 1 mask)
- Region control: Per-region prompts and guidance scales
- Multi-region editing in single denoising pass

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

