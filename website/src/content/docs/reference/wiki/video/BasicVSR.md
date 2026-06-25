---
title: "BasicVSR<T>"
description: "BasicVSR: baseline bidirectional recurrent video super-resolution."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Video.Enhancement`

BasicVSR: baseline bidirectional recurrent video super-resolution.

## For Beginners

BasicVSR processes video frames like reading a book forward and
backward simultaneously. By looking in both directions, each frame gets context from
past AND future frames. Optical flow tells the model how pixels moved between frames,
so it can properly align them before combining their information.

**Usage:**

## How It Works

BasicVSR (Chan et al., CVPR 2021) establishes the essential components for video SR:

- Bidirectional recurrent propagation: processes frames both forward and backward in time,

so each frame benefits from information across the entire sequence

- Optical flow-based alignment: SpyNet estimates motion between adjacent frames and warps

features to compensate for motion before aggregation

- Residual feature refinement: 30 residual blocks per direction refine aligned features
- Pixel shuffle upsampling: sub-pixel convolution for efficient 4x spatial upscaling

BasicVSR serves as the foundation for IconVSR, BasicVSR++, and RealBasicVSR.

**Reference:** "BasicVSR: The Search for Essential Components in Video Super-Resolution
and Beyond" (Chan et al., CVPR 2021)

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `BasicVSR(NeuralNetworkArchitecture<>,BasicVSROptions,IGradientBasedOptimizer<,Tensor<>,Tensor<>>)` | Creates a BasicVSR model in native training mode. |
| `BasicVSR(NeuralNetworkArchitecture<>,String,BasicVSROptions)` | Creates a BasicVSR model in ONNX inference mode. |

## Methods

| Method | Summary |
|:-----|:--------|
| `Upscale(Tensor<>)` |  |

