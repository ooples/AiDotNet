---
title: "FLAVR<T>"
description: "FLAVR: flow-agnostic video representations for fast frame interpolation."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Video.FrameInterpolation`

FLAVR: flow-agnostic video representations for fast frame interpolation.

## For Beginners

Most frame interpolation methods first figure out how objects move
(optical flow), then use that to warp frames. FLAVR skips the flow step entirely by using
3D convolutions that "see" multiple frames at once and directly paint the intermediate
frame. This makes it faster and avoids ghosting artifacts from bad flow estimates.

**Usage:**

## How It Works

FLAVR (Kalluri et al., CVPR 2023) uses 3D convolutions for flow-free interpolation:

- 3D spatio-temporal convolutions: processes multiple input frames simultaneously using

3D (space + time) convolutions that capture temporal relationships without explicit
optical flow estimation, avoiding flow-related artifacts entirely

- 3D encoder-decoder: a U-Net style architecture where the encoder uses strided 3D

convolutions to downsample in both space and time, and the decoder uses transposed
3D convolutions to upsample back to full resolution

- Multi-frame input: takes 4 input frames (2 before and 2 after the target) for richer

temporal context, unlike 2-frame methods that miss longer-range motion patterns

- Direct synthesis: directly outputs the target frame pixels without intermediate flow

or warping operations, avoiding flow estimation errors entirely

**Reference:** "FLAVR: Flow-Agnostic Video Representations for Fast Frame Interpolation"
(Kalluri et al., CVPR 2023)

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `FLAVR(NeuralNetworkArchitecture<>,FLAVROptions,IGradientBasedOptimizer<,Tensor<>,Tensor<>>)` | Creates a FLAVR model in native training mode. |
| `FLAVR(NeuralNetworkArchitecture<>,String,FLAVROptions)` | Creates a FLAVR model in ONNX inference mode. |

## Methods

| Method | Summary |
|:-----|:--------|
| `Interpolate(Tensor<>,Tensor<>,Double)` |  |

