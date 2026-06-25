---
title: "PASDModel<T>"
description: "PASD: Pixel-Aware Stable Diffusion for real-world image super-resolution."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Diffusion.SuperResolution`

PASD: Pixel-Aware Stable Diffusion for real-world image super-resolution.

## For Beginners

When upscaling images, it's crucial to keep the original
structure intact while adding detail. PASD weaves the low-resolution image information
directly into the diffusion model's attention mechanism, ensuring the output matches
the input's structure perfectly while adding sharp, realistic details.

## How It Works

PASD uses pixel-aware cross-attention to inject low-resolution structural information
directly into the diffusion U-Net's attention layers. This ensures the super-resolved
image preserves the original structure while generating realistic high-frequency details.

Technical specifications:

- Architecture: SD2.1 U-Net with pixel-aware cross-attention
- Text encoder: OpenCLIP ViT-H/14 (1024-dim) — uses image captions for guidance
- Pixel-aware attention: LR features injected into cross-attention at every layer
- Scale factor: 4x upscaling
- Personalized stylization also supported

Reference: Yang et al., "Pixel-Aware Stable Diffusion for Realistic Image Super-resolution and Personalized Stylization", 2024

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

