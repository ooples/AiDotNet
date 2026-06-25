---
title: "HDPainterModel<T>"
description: "HD-Painter model for high-resolution inpainting with Prompt-Aware Introverted Attention (PAIntA)."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Diffusion.ImageEditing`

HD-Painter model for high-resolution inpainting with Prompt-Aware Introverted Attention (PAIntA).

## For Beginners

HD-Painter is designed for high-resolution inpainting. When
you inpaint at resolutions larger than what the model was trained on (e.g., 2048x2048),
regular models often produce artifacts. HD-Painter solves this with special attention
mechanisms that work well at any resolution.

## How It Works

HD-Painter addresses resolution limitations in standard inpainting by using
Prompt-Aware Introverted Attention (PAIntA) that constrains self-attention to masked
regions while using text guidance to steer content generation. This enables coherent
inpainting at resolutions beyond the model's native training resolution.

Technical specifications:

- Base model: Stable Diffusion 1.5 inpainting (with PAIntA modification)
- Text encoder: CLIP ViT-L/14 (768-dim)
- Input channels: 9 (4 latent + 4 masked image latent + 1 mask)
- PAIntA: Constrains self-attention to masked region with text guidance
- Supports resolutions beyond training (2K+)

Reference: Manukyan et al., "HD-Painter: High-Resolution and Prompt-Faithful Text-Guided Image Inpainting with Diffusion Models", 2024

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

