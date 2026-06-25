---
title: "AVIDOptions"
description: "Configuration options for the AVID (Audio-Visual Inpainting Diffusion) video inpainting model."
section: "API Reference"
---

`Options & Configuration` · `AiDotNet.Video.Options`

Configuration options for the AVID (Audio-Visual Inpainting Diffusion) video inpainting model.

## For Beginners

AVID options control adaptive video inpainting via diffusion models.

## How It Works

**References:**

- Paper: "AVID: Any-Length Video Inpainting with Diffusion Model" (Zhang et al., CVPR 2024)

AVID uses a diffusion-based approach for video inpainting, supporting arbitrary video lengths
through an autoregressive temporal pipeline with overlapping windows for consistency.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `AVIDOptions` | Initializes a new instance with default values. |
| `AVIDOptions(AVIDOptions)` | Initializes a new instance by copying from another instance. |

## Properties

| Property | Summary |
|:-----|:--------|
| `DropoutRate` | Dropout rate for regularization. |
| `LearningRate` | Learning rate for training. |
| `ModelPath` | Path to the ONNX model file for inference mode. |
| `NumDiffusionSteps` | Number of diffusion steps for the denoising process. |
| `NumFeatures` | Number of base feature channels in the diffusion U-Net. |
| `NumHeads` | Number of attention heads in the temporal transformer layers. |
| `NumResBlocks` | Number of residual blocks per U-Net level. |
| `OnnxOptions` | ONNX runtime options for inference mode. |
| `TemporalOverlap` | Size of overlapping temporal window for long video processing. |
| `Variant` | Model variant controlling capacity and speed trade-off. |

