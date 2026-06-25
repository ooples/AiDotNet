---
title: "VideoP2PModel<T>"
description: "VideoP2P prompt-to-prompt cross-attention control for video editing."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Diffusion.Video.VideoEditing`

VideoP2P prompt-to-prompt cross-attention control for video editing.

## For Beginners

VideoP2P extends prompt-to-prompt image editing to video by controlling cross-attention maps across frames. You can change specific elements (e.g., cat to dog) while preserving the rest of the video content.

## How It Works

**References:**

- Paper: "Video-P2P: Video Editing with Cross-attention Control" (Liu et al., 2024)

VideoP2P extends the Prompt-to-Prompt image editing approach to video by manipulating cross-
attention maps in the diffusion UNet during video generation. Enables precise semantic editing
(e.g., changing object attributes) while preserving temporal consistency through shared attention.

Technical specifications:

- Architecture: Cross-Attention Control + Temporal Consistency
- Latent channels: 4
- Default: 24 frames at 8 FPS
- Supports I2V: No | T2V: Yes | V2V: Yes

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `VideoP2PModel(NeuralNetworkArchitecture<>,DiffusionModelOptions<>,INoiseScheduler<>,VideoUNetPredictor<>,TemporalVAE<>,IConditioningModule<>,Int32,Int32,Nullable<Int32>)` | Initializes a new instance of VideoP2PModel with full customization support. |

