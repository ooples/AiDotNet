---
title: "TurboEditModel<T>"
description: "TurboEdit model for fast few-step image editing using distilled SDXL Turbo."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Diffusion.ImageEditing`

TurboEdit model for fast few-step image editing using distilled SDXL Turbo.

## For Beginners

TurboEdit is fast image editing — it can modify images in
just 3-5 denoising steps, making it nearly instant. Great for interactive editing
workflows where you want to see changes in real time.

## How It Works

TurboEdit adapts DDPM inversion techniques to work with few-step distilled models
(SDXL Turbo), enabling real-time image editing in just 3-5 steps. Uses
carefully calibrated noise injection to maintain edit quality at low step counts.

Technical specifications:

- Base model: SDXL Turbo (adversarially distilled)
- Text encoders: CLIP ViT-L/14 + OpenCLIP ViT-G/14 (2048 context)
- Editing: Adapted DDPM inversion for few-step regime
- Typical steps: 3-5 (vs 20-50 for standard models)
- Input channels: 8 (4 latent + 4 source image latent)
- Guidance: Low (2.0) due to distillation

Reference: Deutch et al., "TurboEdit: Text-Based Image Editing Using Few-Step Diffusion Models", 2024

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

