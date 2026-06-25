---
title: "FuStaOptions"
description: "Configuration options for the FuSta (Full-frame Stabilization) video stabilization model."
section: "API Reference"
---

`Options & Configuration` · `AiDotNet.Video.Options`

Configuration options for the FuSta (Full-frame Stabilization) video stabilization model.

## For Beginners

FuSta options configure the fusion-based video stabilization model.

## How It Works

**References:**

- Paper: "FuSta: Hybrid Approach for Full-frame Video Stabilization" (Liu et al., 2021)

FuSta combines optical flow-based warping with outpainting to achieve full-frame stabilization
without cropping, using a two-stage pipeline: motion compensation + content completion.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `FuStaOptions` | Initializes a new instance with default values. |
| `FuStaOptions(FuStaOptions)` | Initializes a new instance by copying from another instance. |

## Properties

| Property | Summary |
|:-----|:--------|
| `DropoutRate` | Dropout rate for regularization. |
| `LearningRate` | Learning rate for training. |
| `ModelPath` | Path to the ONNX model file for inference mode. |
| `NumFeatures` | Number of base feature channels in the stabilization network. |
| `NumHeads` | Number of attention heads for the content completion transformer. |
| `NumLevels` | Number of encoder-decoder levels in the outpainting branch. |
| `NumResBlocks` | Number of residual blocks in each encoder-decoder level. |
| `OnnxOptions` | ONNX runtime options for inference mode. |
| `Variant` | Model variant controlling capacity and speed trade-off. |

