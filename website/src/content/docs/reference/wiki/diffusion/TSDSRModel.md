---
title: "TSDSRModel<T>"
description: "TSD-SR: Timestep-Shifted Diffusion for fast and high-quality super-resolution."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Diffusion.SuperResolution`

TSD-SR: Timestep-Shifted Diffusion for fast and high-quality super-resolution.

## For Beginners

For super-resolution, you don't need to start from pure noise
— you already have the low-res image. TSD-SR is smart about this: it only adds a
little noise and removes it in fewer steps, making SR 5-10x faster than using a
standard diffusion model for upscaling.

## How It Works

TSD-SR shifts the diffusion timestep range for SR tasks, recognizing that SR needs
less noise removal than generation from scratch. By starting from lower noise levels
and using an adapted schedule, it achieves high-quality 4x upscaling in just 4-10
diffusion steps.

Technical specifications:

- Architecture: SD2.1 U-Net with timestep-shifted schedule
- Text encoder: OpenCLIP ViT-H/14 (1024-dim)
- Timestep shifting: Starts from lower noise levels (t_max ~ 200-400 vs 1000)
- Optimal steps: 4-10 (vs 50+ for standard diffusion SR)
- Speed improvement: 5-10x faster than standard diffusion SR
- Scale factor: 4x upscaling

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

