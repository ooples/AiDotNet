---
title: "Veo3Model<T>"
description: "Veo 3 with native audio generation and dialogue synchronization."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Diffusion.Video`

Veo 3 with native audio generation and dialogue synchronization.

## For Beginners

Veo 3 from Google DeepMind generates videos with native audio including dialogue synchronization. It produces high-fidelity video with matching soundtracks, ambient audio, and character dialogue.

## How It Works

**References:**

- Reference: Google DeepMind Veo 3 (2025)

Veo 3 is the only major model that generates fully synchronized audio including dialogue, music,
and ambient sound directly from a text prompt. It produces cinematic-quality videos optimized for
realism and motion consistency with native audio generation capabilities.

Technical specifications:

- Architecture: DiT + Native Audio Synthesis + Dialogue Sync
- Latent channels: 16
- Default: 240 frames at 24 FPS
- Supports I2V: Yes | T2V: Yes | V2V: No

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `Veo3Model(NeuralNetworkArchitecture<>,DiffusionModelOptions<>,INoiseScheduler<>,DiTNoisePredictor<>,TemporalVAE<>,IConditioningModule<>,Int32,Int32)` | Initializes a new instance of Veo3Model with full customization support. |

