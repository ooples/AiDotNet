---
title: "LumiereModel<T>"
description: "Lumiere Space-Time UNet for single-pass 80-frame video generation."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Diffusion.Video`

Lumiere Space-Time UNet for single-pass 80-frame video generation.

## For Beginners

Lumiere from Google uses a Space-Time UNet (STUNet) to generate entire video sequences in a single pass rather than frame-by-frame. This produces globally consistent motion across all 80 frames simultaneously.

## How It Works

**References:**

- Paper: "Lumiere: A Space-Time Diffusion Model for Video Generation" (Google, 2024)

Lumiere uses a novel Space-Time UNet (STUNet) architecture that downsamples in both space and
time, generating 80 frames at 16 FPS in a single pass. This holistic approach produces more
globally coherent motion compared to cascaded models that generate keyframes then interpolate.

Technical specifications:

- Architecture: STUNet (Space-Time UNet) + Single-Pass Generation
- Latent channels: 4
- Default: 80 frames at 16 FPS
- Supports I2V: Yes | T2V: Yes | V2V: No

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `LumiereModel(NeuralNetworkArchitecture<>,DiffusionModelOptions<>,INoiseScheduler<>,VideoUNetPredictor<>,TemporalVAE<>,IConditioningModule<>,Int32,Int32)` | Initializes a new instance of LumiereModel with full customization support. |

