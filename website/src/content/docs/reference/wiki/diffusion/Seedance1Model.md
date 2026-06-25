---
title: "Seedance1Model<T>"
description: "Seedance 1 ranked #1 on T2V and I2V leaderboards."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Diffusion.Video`

Seedance 1 ranked #1 on T2V and I2V leaderboards.

## For Beginners

Seedance 1 from ByteDance ranks first on video generation leaderboards for both text-to-video and image-to-video. It produces highly detailed, temporally consistent videos with excellent prompt adherence.

## How It Works

**References:**

- Reference: ByteDance Seedance 1 (2025)

Seedance 1 from ByteDance achieved #1 ranking on the Artificial Analysis T2V and I2V leaderboards.
The model combines efficient DiT architecture with advanced temporal modeling and precise text
understanding for state-of-the-art video generation quality.

Technical specifications:

- Architecture: Efficient DiT + Advanced Temporal Modeling
- Latent channels: 16
- Default: 120 frames at 24 FPS
- Supports I2V: Yes | T2V: Yes | V2V: No

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `Seedance1Model(NeuralNetworkArchitecture<>,DiffusionModelOptions<>,INoiseScheduler<>,DiTNoisePredictor<>,TemporalVAE<>,IConditioningModule<>,Int32,Int32)` | Initializes a new instance of Seedance1Model with full customization support. |

