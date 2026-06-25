---
title: "CycleGANTurboModel<T>"
description: "CycleGAN-Turbo model combining CycleGAN unpaired translation with a diffusion backbone."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Diffusion.ImageEditing`

CycleGAN-Turbo model combining CycleGAN unpaired translation with a diffusion backbone.

## For Beginners

CycleGAN-Turbo translates images between two styles (like
photos to paintings, summer to winter) using a single fast step. Unlike traditional
CycleGAN, it leverages a pre-trained SDXL model for much higher quality results.

## How It Works

Combines the unpaired image-to-image translation paradigm of CycleGAN with a
pre-trained SDXL diffusion model backbone. Uses cycle consistency loss on a diffusion
U-Net to enable single-step unpaired domain translation with high fidelity.

Technical specifications:

- Base model: SDXL U-Net with cycle consistency fine-tuning
- Text encoders: CLIP ViT-L/14 + OpenCLIP ViT-G/14 (2048 context)
- Input channels: 8 (4 source latent + 4 target latent)
- Inference: Single-step (1 NFE)
- Guidance: 1.0 (single-step, no CFG needed)
- Training: Cycle consistency + adversarial loss on SDXL

Reference: Parmar et al., "One-Step Image Translation with Text-to-Image Models", 2024

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

