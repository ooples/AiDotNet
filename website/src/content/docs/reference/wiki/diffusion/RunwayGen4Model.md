---
title: "RunwayGen4Model<T>"
description: "Runway Gen-4 multi-modal understanding and generation model."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Diffusion.Video`

Runway Gen-4 multi-modal understanding and generation model.

## For Beginners

Runway Gen-4 combines scene understanding with video generation, enabling multi-modal control over the output. It can generate from text, images, or existing video with fine-grained control over style, motion, and composition.

## How It Works

**References:**

- Reference: Runway Gen-4 (2025)

Runway Gen-4 combines multi-modal understanding with video generation, enabling the model to
understand and generate video from diverse inputs including text, images, video references, and
style descriptions. Features advanced camera control and consistent character generation.

Technical specifications:

- Architecture: Multi-Modal DiT + Understanding Engine
- Latent channels: 16
- Default: 120 frames at 24 FPS
- Supports I2V: Yes | T2V: Yes | V2V: No

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `RunwayGen4Model(NeuralNetworkArchitecture<>,DiffusionModelOptions<>,INoiseScheduler<>,DiTNoisePredictor<>,TemporalVAE<>,IConditioningModule<>,Int32,Int32)` | Initializes a new instance of RunwayGen4Model with full customization support. |

