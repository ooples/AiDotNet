---
title: "FLAVROptions"
description: "Configuration options for the FLAVR flow-agnostic video representations model."
section: "API Reference"
---

`Options & Configuration` · `AiDotNet.Video.Options`

Configuration options for the FLAVR flow-agnostic video representations model.

## For Beginners

Most frame interpolation methods first figure out how objects move
(optical flow), then use that to warp frames. FLAVR skips the flow step entirely by using
3D convolutions that "see" multiple frames at once and directly paint the intermediate
frame. This makes it faster and avoids the ghosting artifacts that come from bad flow
estimates, especially in scenes with transparent objects or repetitive patterns.

## How It Works

FLAVR (Kalluri et al., CVPR 2023) uses 3D convolutions for flow-free interpolation:

- 3D spatio-temporal convolutions: processes multiple input frames simultaneously using

3D (space + time) convolutions that capture temporal relationships without explicit
optical flow estimation

- 3D encoder-decoder: a U-Net style architecture where the encoder uses strided 3D

convolutions to downsample in both space and time, and the decoder uses transposed
3D convolutions to upsample back to full resolution

- Multi-frame input: takes 4 input frames (2 before and 2 after the target) for richer

temporal context, unlike 2-frame methods

- Direct synthesis: directly outputs the target frame pixels without intermediate flow

or warping operations, avoiding flow estimation errors entirely

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `FLAVROptions` | Initializes a new instance with default values. |
| `FLAVROptions(FLAVROptions)` | Initializes a new instance by copying from another instance. |

## Properties

| Property | Summary |
|:-----|:--------|
| `DropoutRate` | Gets or sets the dropout rate. |
| `LearningRate` | Gets or sets the learning rate. |
| `ModelPath` | Gets or sets the path to the ONNX model file. |
| `NumFeatures` | Gets or sets the number of feature channels. |
| `NumInputFrames` | Gets or sets the number of input frames (temporal context window). |
| `NumLevels` | Gets or sets the number of encoder/decoder levels. |
| `NumResBlocks` | Gets or sets the number of 3D residual blocks per encoder level. |
| `OnnxOptions` | Gets or sets the ONNX runtime options. |
| `TemporalKernelSize` | Gets or sets the temporal kernel size for 3D convolutions. |
| `Variant` | Gets or sets the model variant. |

