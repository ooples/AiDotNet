---
title: "M2MOptions"
description: "Configuration options for M2M many-to-many splatting frame interpolation."
section: "API Reference"
---

`Options & Configuration` · `AiDotNet.Video.Options`

Configuration options for M2M many-to-many splatting frame interpolation.

## For Beginners

Standard interpolation "pulls" each pixel from one location in the
source frame. M2M instead "pushes" pixels from the source to potentially multiple locations
in the target, which better handles cases where objects overlap or appear/disappear between
frames.

## How It Works

M2M (Hu et al., CVPR 2022) uses many-to-many splatting for efficient interpolation:

- Many-to-many splatting: instead of the standard one-to-one backward warping (each target

pixel samples from one source pixel), M2M allows multiple source pixels to contribute to
multiple target pixels simultaneously using forward splatting with learned weights

- Multiple bidirectional flows: estimates K flow field pairs (forward and backward) at each

pyramid level, capturing multiple motion hypotheses for occluded regions and motion
boundaries where a single flow is ambiguous

- Splatting confidence: each splatted pixel carries a learned confidence weight, and the

final pixel value is a confidence-weighted sum of all contributions, naturally handling
occlusions and disocclusions

- Multi-scale pipeline: coarse-to-fine architecture where splatting is performed at each

scale, and residual corrections are added at each level

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `M2MOptions` | Initializes a new instance with default values. |
| `M2MOptions(M2MOptions)` | Initializes a new instance by copying from another instance. |

## Properties

| Property | Summary |
|:-----|:--------|
| `DropoutRate` | Gets or sets the dropout rate. |
| `LearningRate` | Gets or sets the learning rate. |
| `ModelPath` | Gets or sets the path to the ONNX model file. |
| `NumFeatures` | Gets or sets the number of feature channels. |
| `NumFlowHypotheses` | Gets or sets the number of flow hypotheses per pixel (K). |
| `NumPyramidLevels` | Gets or sets the number of pyramid levels for multi-scale splatting. |
| `NumRefineBlocks` | Gets or sets the number of refinement blocks at each scale level. |
| `OnnxOptions` | Gets or sets the ONNX runtime options. |
| `SplattingRadius` | Gets or sets the splatting radius in pixels. |
| `Variant` | Gets or sets the model variant. |

