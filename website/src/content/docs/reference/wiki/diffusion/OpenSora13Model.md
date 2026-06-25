---
title: "OpenSora13Model<T>"
description: "Open-Sora 1.3 with upgraded 3D-VAE and rectified flow."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Diffusion.Video`

Open-Sora 1.3 with upgraded 3D-VAE and rectified flow.

## For Beginners

Open-Sora 1.3 is a fully open-source video generation model that supports any resolution and aspect ratio. It uses alternating spatial and temporal attention (STDiT) for efficiency and rectified flow training for stable, high-quality generation from text prompts.

## How It Works

**References:**

- Paper: "Open-Sora: Democratizing Efficient Video Production for All" (HPC-AI Tech, 2024)

Open-Sora 1.3 features an upgraded 3D-VAE with improved temporal compression and rectified flow
training objective. The STDiT backbone alternates spatial and temporal attention for efficient
video generation. Supports any resolution and aspect ratio through dynamic patching.

Technical specifications:

- Architecture: STDiT + Rectified Flow + Upgraded 3D-VAE
- Latent channels: 4
- Default: 51 frames at 24 FPS
- Supports I2V: Yes | T2V: Yes | V2V: No

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `OpenSora13Model(NeuralNetworkArchitecture<>,DiffusionModelOptions<>,INoiseScheduler<>,DiTNoisePredictor<>,TemporalVAE<>,IConditioningModule<>,Int32,Int32,Nullable<Int32>)` | Initializes a new instance of OpenSora13Model with full customization support. |

