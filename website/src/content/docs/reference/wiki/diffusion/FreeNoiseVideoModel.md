---
title: "FreeNoiseVideoModel<T>"
description: "FreeNoise extended video generation through noise rescheduling."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Diffusion.Video.LongVideo`

FreeNoise extended video generation through noise rescheduling.

## For Beginners

FreeNoise extends short video generators to produce longer sequences by intelligently rescheduling the noise patterns. This simple technique enables existing models to generate longer videos without retraining.

## How It Works

**References:**

- Paper: "FreeNoise: Tuning-Free Longer Video Diffusion via Noise Rescheduling" (2024)

FreeNoise enables longer video generation without additional training by rescheduling the noise
initialization. Instead of independent noise for each frame, it uses temporally correlated noise
with a sliding window scheme. This produces longer, more coherent videos from any short-video
diffusion model.

Technical specifications:

- Architecture: Noise Rescheduling + Sliding Window + Temporal Correlation
- Latent channels: 4
- Default: 64 frames at 8 FPS
- Supports I2V: Yes | T2V: Yes | V2V: No

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `FreeNoiseVideoModel(NeuralNetworkArchitecture<>,DiffusionModelOptions<>,INoiseScheduler<>,VideoUNetPredictor<>,TemporalVAE<>,IConditioningModule<>,Int32,Int32)` | Initializes a new instance of FreeNoiseVideoModel with full customization support. |

