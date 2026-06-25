---
title: "PowerPaintModel<T>"
description: "PowerPaint v2 model for versatile task-aware image inpainting with learnable task prompts."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Diffusion.ImageEditing`

PowerPaint v2 model for versatile task-aware image inpainting with learnable task prompts.

## For Beginners

PowerPaint is a versatile inpainting model that can handle
many different tasks: removing objects, inserting new objects, changing shapes, or
extending images. You just tell it which task you want, and it adapts automatically
through special learned task tokens.

## How It Works

PowerPaint uses learnable task prompts to handle multiple inpainting tasks including
text-guided object insertion, object removal, shape-guided generation, and outpainting
with a single model. Task-specific prompt tokens are prepended to guide behavior.
Built on Stable Diffusion 2.1 with OpenCLIP ViT-H text encoder.

Technical specifications:

- Base model: Stable Diffusion 2.1 (UNet)
- Text encoder: OpenCLIP ViT-H/14 (1024-dim context)
- Input channels: 9 (4 latent + 4 masked image + 1 mask)
- Task prompts: 4 learnable tokens per task type
- Tasks: text-guided insertion, removal, shape-guided, outpainting
- Resolution: 512x512 native

Reference: Zhuang et al., "A Task is Worth One Word: Learning with Task Prompts for High-Quality Versatile Image Inpainting", 2024

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

