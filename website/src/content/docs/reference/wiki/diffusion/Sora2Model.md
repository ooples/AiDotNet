---
title: "Sora2Model<T>"
description: "Sora 2 cinematic video generation with physics simulation and synced audio."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Diffusion.Video`

Sora 2 cinematic video generation with physics simulation and synced audio.

## For Beginners

Sora 2 is OpenAI's advanced video generation model producing cinematic-quality videos with realistic physics simulation and synchronized audio. It understands 3D scene consistency, camera motion, and can generate videos up to a minute long.

## How It Works

**References:**

- Reference: OpenAI Sora 2 (2025)

Sora 2 generates cinematic-quality videos with realistic physics simulation, synchronized audio,
and strong prompt adherence. The model demonstrates understanding of cause-and-effect relationships
and can produce videos with natural lighting, reflections, and physical interactions.

Technical specifications:

- Architecture: DiT + Physics Simulation + Audio Sync
- Latent channels: 16
- Default: 240 frames at 24 FPS
- Supports I2V: Yes | T2V: Yes | V2V: No

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `Sora2Model(NeuralNetworkArchitecture<>,DiffusionModelOptions<>,INoiseScheduler<>,DiTNoisePredictor<>,TemporalVAE<>,IConditioningModule<>,Int32,Int32,Nullable<Int32>)` | Initializes a new instance of Sora2Model with full customization support. |

