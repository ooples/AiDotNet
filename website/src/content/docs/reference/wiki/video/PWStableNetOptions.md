---
title: "PWStableNetOptions"
description: "Configuration options for the PWStableNet (Pixel-Wise Stable Net) video stabilization model."
section: "API Reference"
---

`Options & Configuration` · `AiDotNet.Video.Options`

Configuration options for the PWStableNet (Pixel-Wise Stable Net) video stabilization model.

## For Beginners

PWStableNet options configure the pixel-wise video stabilization model.

## How It Works

**References:**

- Paper: "PWStableNet: Learning Pixel-Wise Warping Maps for Video Stabilization" (Zhao et al., IEEE TIP 2020)

PWStableNet predicts per-pixel warping maps (not global homographies) for more flexible
stabilization that handles parallax and rolling shutter distortion.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `PWStableNetOptions` | Initializes a new instance with default values. |
| `PWStableNetOptions(PWStableNetOptions)` | Initializes a new instance by copying from another instance. |

## Properties

| Property | Summary |
|:-----|:--------|
| `DropoutRate` | Dropout rate for regularization. |
| `GridSize` | Grid size for the spatial transformer network that produces warp fields. |
| `LearningRate` | Learning rate for training. |
| `ModelPath` | Path to the ONNX model file for inference mode. |
| `NumFeatures` | Number of base feature channels. |
| `NumRefinementIters` | Number of coarse-to-fine refinement iterations for the warp field. |
| `NumResBlocks` | Number of residual blocks in the feature extraction backbone. |
| `OnnxOptions` | ONNX runtime options for inference mode. |
| `Variant` | Model variant controlling capacity and speed trade-off. |

