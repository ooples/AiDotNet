---
title: "OpenSora2Model<T>"
description: "Open-Sora 2.0 commercial-level video generation model."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Diffusion.Video`

Open-Sora 2.0 commercial-level video generation model.

## For Beginners

Open-Sora 2.0 is a commercial-grade open-source video generator trained for under $200K. It produces high-quality videos competitive with proprietary models, using an improved architecture with better temporal modeling and data curation.

## How It Works

**References:**

- Paper: "Open-Sora 2.0: Training a Commercial-Level Video Generation Model in $200k" (HPC-AI Tech, 2025)

Open-Sora 2.0 achieves commercial-level video quality comparable to HunyuanVideo and Runway Gen-3
alpha while being trained for only $200k. Uses an improved STDiT backbone with enhanced temporal
attention, better VAE, and data curation pipeline. Human evaluation shows it matches or exceeds
leading commercial models.

Technical specifications:

- Architecture: Enhanced STDiT + Improved 3D-VAE + Rectified Flow
- Latent channels: 16
- Default: 93 frames at 24 FPS
- Supports I2V: Yes | T2V: Yes | V2V: No

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `OpenSora2Model(NeuralNetworkArchitecture<>,DiffusionModelOptions<>,INoiseScheduler<>,DiTNoisePredictor<>,TemporalVAE<>,IConditioningModule<>,Int32,Int32,Nullable<Int32>)` | Initializes a new instance of OpenSora2Model with full customization support. |

