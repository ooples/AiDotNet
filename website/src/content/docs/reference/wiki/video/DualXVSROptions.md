---
title: "DualXVSROptions"
description: "Configuration options for the DualX-VSR dual axial spatial-temporal transformer."
section: "API Reference"
---

`Options & Configuration` · `AiDotNet.Video.Options`

Configuration options for the DualX-VSR dual axial spatial-temporal transformer.

## For Beginners

Most video SR models need to figure out how objects moved between
frames (optical flow). DualX-VSR skips this step entirely by using a clever attention
pattern that looks along two crossing axes simultaneously, naturally capturing motion
without explicit computation. Think of it like looking at a crossword puzzle -- by
reading both across and down, you understand the full picture.

## How It Works

DualX-VSR (2025) eliminates explicit motion compensation through dual axial attention:

- Dual axial attention: decomposes 3D (H x W x T) attention into two orthogonal axes

-- spatial-height-temporal and spatial-width-temporal -- reducing complexity from
O((HWT)^2) to O(HWT * max(H,W,T))

- Motion-free alignment: the dual axial attention implicitly captures inter-frame

correspondence without computing optical flow or deformable offsets

- Symmetric bidirectional processing: forward and backward temporal propagation with

shared axial attention weights

- Efficient design: linear complexity in the number of frames while maintaining

full spatial-temporal receptive field

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `DualXVSROptions` | Initializes a new instance with default values. |
| `DualXVSROptions(DualXVSROptions)` | Initializes a new instance by copying from another instance. |

## Properties

| Property | Summary |
|:-----|:--------|
| `DropoutRate` | Gets or sets the dropout rate. |
| `LearningRate` | Gets or sets the learning rate. |
| `ModelPath` | Gets or sets the path to the ONNX model file. |
| `NumAxialBlocks` | Gets or sets the number of dual axial transformer blocks. |
| `NumFeatures` | Gets or sets the number of feature channels. |
| `NumHeads` | Gets or sets the number of attention heads for axial attention. |
| `OnnxOptions` | Gets or sets the ONNX runtime options. |
| `ScaleFactor` | Gets or sets the spatial upscaling factor. |
| `TemporalWindow` | Gets or sets the temporal window size for axial attention. |
| `Variant` | Gets or sets the model variant. |
| `WeightDecay` | Gets or sets the decoupled weight decay for AdamW training. |

