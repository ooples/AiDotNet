---
title: "MAGI1Model<T>"
description: "MAGI-1 video model with strong temporal coherence and multi-task support."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Diffusion.Video`

MAGI-1 video model with strong temporal coherence and multi-task support.

## For Beginners

MAGI-1 is a ~10B parameter video model with strong temporal coherence, meaning objects and people stay consistent across frames. It supports multiple tasks (text-to-video, image-to-video) and produces smooth, natural-looking motion.

## How It Works

**References:**

- Paper: "MAGI-1: Autoregressive Video Generation at Scale" (Sand AI, 2025)

MAGI-1 is a ~10B parameter model from Sand AI achieving strong temporal coherence through
autoregressive generation with latent space consistency mechanisms. It supports multiple tasks
including text-to-video, image-to-video, and video continuation. The model uses a DiT backbone
with enhanced temporal modeling.

Technical specifications:

- Architecture: Autoregressive DiT + Temporal Coherence
- Latent channels: 16
- Default: 65 frames at 24 FPS
- Supports I2V: Yes | T2V: Yes | V2V: No

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `MAGI1Model(NeuralNetworkArchitecture<>,DiffusionModelOptions<>,INoiseScheduler<>,DiTNoisePredictor<>,TemporalVAE<>,IConditioningModule<>,Int32,Int32)` | Initializes a new instance of MAGI1Model with full customization support. |

