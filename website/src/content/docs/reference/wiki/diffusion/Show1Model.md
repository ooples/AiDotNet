---
title: "Show1Model<T>"
description: "Show-1 marrying pixel and latent diffusion for text-to-video."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Diffusion.Video.LongVideo`

Show-1 marrying pixel and latent diffusion for text-to-video.

## For Beginners

Show-1 combines pixel-space and latent-space diffusion in a two-stage pipeline. The first stage generates coarse video in pixel space, then the second refines it in latent space for high-quality output.

## How It Works

**References:**

- Paper: "Show-1: Marrying Pixel and Latent Diffusion Models for Text-to-Video Generation" (2023)

Show-1 combines pixel-space and latent-space diffusion models in a cascaded pipeline. The pixel
diffusion model handles low-resolution generation for accurate motion, while the latent diffusion
model performs super-resolution for visual quality. This marriage leverages the strengths of both
approaches.

Technical specifications:

- Architecture: Pixel Diffusion + Latent Super-Resolution Cascade
- Latent channels: 4
- Default: 29 frames at 8 FPS
- Supports I2V: Yes | T2V: Yes | V2V: No

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `Show1Model(NeuralNetworkArchitecture<>,DiffusionModelOptions<>,INoiseScheduler<>,VideoUNetPredictor<>,TemporalVAE<>,IConditioningModule<>,Int32,Int32,Nullable<Int32>)` | Initializes a new instance of Show1Model with full customization support. |

