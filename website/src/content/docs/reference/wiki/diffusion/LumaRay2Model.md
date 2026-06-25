---
title: "LumaRay2Model<T>"
description: "Luma Ray 2 video model with fast natural motion and better physics."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Diffusion.Video`

Luma Ray 2 video model with fast natural motion and better physics.

## For Beginners

Luma Ray 2 produces videos with fast, natural motion using 10x more compute than its predecessor. It excels at realistic movement patterns and physical interactions between objects.

## How It Works

**References:**

- Reference: Luma AI Ray 2 (2025)

Ray 2 was trained with 10x more compute than Ray1, offering fast, natural coherent motion and
physics understanding. The model generates videos with natural motion dynamics and good temporal
consistency. Features improved physics simulation for realistic object interactions.

Technical specifications:

- Architecture: DiT + Enhanced Physics + 10x Compute
- Latent channels: 16
- Default: 120 frames at 24 FPS
- Supports I2V: Yes | T2V: Yes | V2V: No

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `LumaRay2Model(NeuralNetworkArchitecture<>,DiffusionModelOptions<>,INoiseScheduler<>,DiTNoisePredictor<>,TemporalVAE<>,IConditioningModule<>,Int32,Int32)` | Initializes a new instance of LumaRay2Model with full customization support. |

