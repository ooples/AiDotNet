---
title: "TokenFlowModel<T>"
description: "TokenFlow consistent video editing via token flow propagation."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Diffusion.Video.VideoEditing`

TokenFlow consistent video editing via token flow propagation.

## For Beginners

TokenFlow enables consistent video editing by propagating visual changes through time using token flow, tracking how image features move between frames. This ensures edits look natural and consistent across the entire video.

## How It Works

**References:**

- Paper: "TokenFlow: Consistent Diffusion Features for Consistent Video Editing" (Bar-Tal et al., 2023)

TokenFlow achieves consistent video editing by explicitly enforcing semantic correspondences of
diffusion features across video frames. The method propagates edited tokens from keyframes to
all frames using the natural token flow, maintaining temporal consistency without per-frame
processing. Works with any text-guided image editing method.

Technical specifications:

- Architecture: Token Flow Propagation + DDIM Inversion + Keyframe Editing
- Latent channels: 4
- Default: 24 frames at 8 FPS
- Supports I2V: No | T2V: Yes | V2V: Yes

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `TokenFlowModel(NeuralNetworkArchitecture<>,DiffusionModelOptions<>,INoiseScheduler<>,VideoUNetPredictor<>,TemporalVAE<>,IConditioningModule<>,Int32,Int32)` | Initializes a new instance of TokenFlowModel with full customization support. |

