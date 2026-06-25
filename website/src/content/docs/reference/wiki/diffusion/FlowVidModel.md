---
title: "FlowVidModel<T>"
description: "FlowVid optical flow guided video-to-video synthesis."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Diffusion.Video.VideoEditing`

FlowVid optical flow guided video-to-video synthesis.

## For Beginners

FlowVid uses optical flow (tracking pixel movement between frames) to guide video-to-video synthesis. This ensures that edits follow the natural motion in the original video for temporal consistency.

## How It Works

**References:**

- Paper: "FlowVid: Taming Imperfect Optical Flows for Consistent Video-to-Video Synthesis" (2024)

FlowVid uses optical flow to guide video-to-video synthesis, maintaining temporal consistency by
warping features along motion trajectories. The method handles imperfect optical flows through
a flow-guided attention mechanism that adaptively weighs contributions from neighboring frames.

Technical specifications:

- Architecture: Optical Flow Guidance + Flow-Guided Attention
- Latent channels: 4
- Default: 24 frames at 8 FPS
- Supports I2V: No | T2V: Yes | V2V: Yes

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `FlowVidModel(NeuralNetworkArchitecture<>,DiffusionModelOptions<>,INoiseScheduler<>,VideoUNetPredictor<>,TemporalVAE<>,IConditioningModule<>,Int32,Int32,Nullable<Int32>)` | Initializes a new instance of FlowVidModel with full customization support. |

