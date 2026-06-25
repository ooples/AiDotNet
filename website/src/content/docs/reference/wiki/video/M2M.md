---
title: "M2M<T>"
description: "M2M: many-to-many splatting for efficient video frame interpolation."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Video.FrameInterpolation`

M2M: many-to-many splatting for efficient video frame interpolation.

## For Beginners

Standard interpolation "pulls" each pixel from one location in the
source frame. M2M instead "pushes" pixels from the source to potentially multiple locations
in the target, which better handles cases where objects overlap or appear/disappear between
frames.

**Usage:**

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

**Reference:** "Many-to-many Splatting for Efficient Video Frame Interpolation"
(Hu et al., CVPR 2022)

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `M2M(NeuralNetworkArchitecture<>,M2MOptions,IGradientBasedOptimizer<,Tensor<>,Tensor<>>)` | Creates an M2M model in native training mode. |
| `M2M(NeuralNetworkArchitecture<>,String,M2MOptions)` | Creates an M2M model in ONNX inference mode. |

## Methods

| Method | Summary |
|:-----|:--------|
| `Interpolate(Tensor<>,Tensor<>,Double)` |  |

