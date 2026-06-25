---
title: "ShiftNetOptions"
description: "Configuration options for ShiftNet channel-shifting video denoising."
section: "API Reference"
---

`Options & Configuration` · `AiDotNet.Video.Options`

Configuration options for ShiftNet channel-shifting video denoising.

## For Beginners

ShiftNet uses a clever trick to denoise video: instead of computing
expensive optical flow, it simply shifts some feature channels to come from past or future
frames. This free operation lets the network naturally learn to use temporal information
for denoising without any extra computation cost.

## How It Works

ShiftNet (Maggioni et al., 2021) uses feature-level temporal shifting for denoising:

- Channel shifting: shifts feature channels along the temporal dimension (some channels

come from past frames, some from future) without explicit alignment or flow, providing
zero-cost temporal information exchange

- Shift-and-aggregate: after shifting, local convolutions aggregate the temporally mixed

features, implicitly learning to handle motion without optical flow

- U-Net backbone: standard encoder-decoder with skip connections, where each conv block

incorporates channel shifting for temporal awareness

- Efficient design: no motion estimation or attention needed, making it faster than

flow-based or attention-based temporal methods

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `ShiftNetOptions` | Initializes a new instance with default values. |
| `ShiftNetOptions(ShiftNetOptions)` | Initializes a new instance by copying from another instance. |

## Properties

| Property | Summary |
|:-----|:--------|
| `DropoutRate` | Gets or sets the dropout rate. |
| `LearningRate` | Gets or sets the learning rate. |
| `ModelPath` | Gets or sets the path to the ONNX model file. |
| `NumBlocks` | Gets or sets the number of encoder/decoder blocks. |
| `NumFeatures` | Gets or sets the number of feature channels. |
| `NumShifts` | Gets or sets the number of temporal shifts per block. |
| `OnnxOptions` | Gets or sets the ONNX runtime options. |
| `ShiftRadius` | Gets or sets the temporal radius for shifting. |
| `Variant` | Gets or sets the model variant. |

