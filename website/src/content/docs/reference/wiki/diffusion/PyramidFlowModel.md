---
title: "PyramidFlowModel<T>"
description: "Pyramid Flow multi-resolution video generation via pyramid flow matching."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Diffusion.Video`

Pyramid Flow multi-resolution video generation via pyramid flow matching.

## For Beginners

Pyramid Flow generates videos by starting at low resolution and progressively adding detail, similar to building a pyramid. This multi-resolution approach produces more globally consistent results and is computationally efficient.

## How It Works

**References:**

- Paper: "Pyramid Flow: Multi-Resolution Flow Matching for Video Generation" (Community, 2024)

Pyramid Flow generates videos at multiple resolutions using a pyramid flow matching approach.
Starting from low resolution, each pyramid level adds spatial and temporal detail through
cascaded flow matching. This enables efficient long-form video generation with progressive
quality enhancement.

Technical specifications:

- Architecture: Pyramid DiT + Multi-Resolution Flow Matching
- Latent channels: 16
- Default: 65 frames at 24 FPS
- Supports I2V: Yes | T2V: Yes | V2V: No

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `PyramidFlowModel(NeuralNetworkArchitecture<>,DiffusionModelOptions<>,INoiseScheduler<>,DiTNoisePredictor<>,TemporalVAE<>,IConditioningModule<>,Int32,Int32)` | Initializes a new instance of PyramidFlowModel with full customization support. |

