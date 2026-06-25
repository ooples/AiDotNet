---
title: "StreamingT2VModel<T>"
description: "StreamingT2V autoregressive long video generation up to 1200 frames."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Diffusion.Video.LongVideo`

StreamingT2V autoregressive long video generation up to 1200 frames.

## For Beginners

Streaming T2V generates long videos (up to 2 minutes / 1200 frames) by producing segments autoregressively - each new segment continues seamlessly from the previous one. This overcomes the memory limitations of generating all frames at once.

## How It Works

**References:**

- Paper: "StreamingT2V: Consistent, Dynamic, and Extendable Long Video Generation from Text" (2024)

StreamingT2V enables creation of long videos (up to 1200 frames / 2 minutes) through an advanced
autoregressive technique. It generates short clips and extends them using temporal overlap and
conditioning on previous frames. Ensures rich motion dynamics and temporal consistency throughout
the entire video sequence.

Technical specifications:

- Architecture: Autoregressive UNet + Temporal Overlap + Long Video
- Latent channels: 4
- Default: 1200 frames at 10 FPS
- Supports I2V: Yes | T2V: Yes | V2V: No

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `StreamingT2VModel(NeuralNetworkArchitecture<>,DiffusionModelOptions<>,INoiseScheduler<>,VideoUNetPredictor<>,TemporalVAE<>,IConditioningModule<>,Int32,Int32,Nullable<Int32>)` | Initializes a new instance of StreamingT2VModel with full customization support. |

