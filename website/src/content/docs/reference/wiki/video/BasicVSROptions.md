---
title: "BasicVSROptions"
description: "Configuration options for the BasicVSR bidirectional recurrent video super-resolution model."
section: "API Reference"
---

`Options & Configuration` · `AiDotNet.Video.Options`

Configuration options for the BasicVSR bidirectional recurrent video super-resolution model.

## For Beginners

BasicVSR treats video frames like a sequence. It processes frames
both forward in time and backward, so each frame benefits from information in both
directions. Optical flow tells the model how pixels moved between frames, so it can
align them before combining their features for higher resolution output.

## How It Works

BasicVSR (Chan et al., CVPR 2021) establishes the essential components for video SR:

- Bidirectional recurrent propagation: forward and backward passes across frames
- Optical flow-based alignment: SpyNet estimates motion between adjacent frames
- Residual feature refinement: 30 residual blocks per propagation direction

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `BasicVSROptions` | Initializes a new instance with default values. |
| `BasicVSROptions(BasicVSROptions)` | Initializes a new instance by copying from another instance. |

## Properties

| Property | Summary |
|:-----|:--------|
| `DropoutRate` | Gets or sets the dropout rate. |
| `LearningRate` | Gets or sets the learning rate. |
| `MidChannels` | Gets or sets the mid-channel dimension in residual blocks. |
| `ModelPath` | Gets or sets the path to the ONNX model file. |
| `NumFeatures` | Gets or sets the number of feature channels in the propagation branches. |
| `NumFrames` | Gets or sets the number of input frames for bidirectional propagation. |
| `NumResBlocks` | Gets or sets the number of residual blocks per propagation direction. |
| `OnnxOptions` | Gets or sets the ONNX runtime options. |
| `ScaleFactor` | Gets or sets the spatial upscaling factor. |
| `Variant` | Gets or sets the model variant. |
| `WarmupSteps` | Gets or sets the number of warmup iterations. |

