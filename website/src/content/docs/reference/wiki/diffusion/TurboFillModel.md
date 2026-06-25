---
title: "TurboFillModel<T>"
description: "TurboFill model for fast few-step inpainting using adversarial distillation on SDXL."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Diffusion.ImageEditing`

TurboFill model for fast few-step inpainting using adversarial distillation on SDXL.

## For Beginners

Normal inpainting models need 20-50 steps, which can be slow.
TurboFill has been specially trained to produce good results in just 4-8 steps,
making it much faster — almost real-time for editing workflows.

## How It Works

TurboFill achieves high-quality inpainting in just 4-8 denoising steps by applying
adversarial distillation to an inpainting-finetuned SDXL model. This makes it
suitable for interactive and real-time inpainting applications.

Technical specifications:

- Base model: SDXL Inpainting (adversarially distilled)
- Text encoders: CLIP ViT-L/14 + OpenCLIP ViT-G/14 (2048 context)
- Input channels: 9 (4 latent + 4 masked image latent + 1 mask)
- Typical steps: 4-8 (vs 20-50 for standard SDXL)
- Guidance: Low (2.0) due to adversarial distillation
- Resolution: 1024x1024 native

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

