---
title: "Pika21Model<T>"
description: "Pika 2.1 short-form video with creative effects."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Diffusion.Video`

Pika 2.1 short-form video with creative effects.

## For Beginners

Pika 2.1 specializes in short-form video generation with creative physics effects like crush, inflate, and melt. It is designed for social media content creation with fun, viral-ready video effects.

## How It Works

**References:**

- Reference: Pika Labs Pika 2.1 (2025)

Pika 2.1 specializes in short-form video generation with creative effects like crush, inflate,
melt, and explode. Features scene ingredients for visual consistency across different scenes
and real-time editing capabilities. Optimized for creative and marketing use cases.

Technical specifications:

- Architecture: DiT + Creative Effects Engine
- Latent channels: 16
- Default: 72 frames at 24 FPS
- Supports I2V: Yes | T2V: Yes | V2V: No

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `Pika21Model(NeuralNetworkArchitecture<>,DiffusionModelOptions<>,INoiseScheduler<>,DiTNoisePredictor<>,TemporalVAE<>,IConditioningModule<>,Int32,Int32,Nullable<Int32>)` | Initializes a new instance of Pika21Model with full customization support. |

