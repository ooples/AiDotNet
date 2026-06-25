---
title: "SoftSplat<T>"
description: "SoftSplat: softmax splatting for video frame interpolation."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Video.FrameInterpolation`

SoftSplat: softmax splatting for video frame interpolation.

## For Beginners

SoftSplat uses a smart voting system (softmax) where each pixel gets
a learned "importance score" to decide which pixel wins when there's a conflict at the same
target position, naturally handling which objects appear in front of others.

**Usage:**

## How It Works

SoftSplat (Niklaus & Liu, CVPR 2020) uses softmax splatting for forward warping:

- Forward warping with softmax: source pixels are "splatted" to target positions, with

conflicts resolved via softmax weighting instead of backward warping

- Importance metric Z: each source pixel carries a learned importance metric Z that controls

its softmax weight, automatically learning foreground/background occlusion ordering

- Feature-space splatting: splatting is performed on deep feature maps rather than raw

pixels, providing richer representations for the synthesis network

- GridNet synthesis: a GridNet-style synthesis network takes splatted features and produces

the final interpolated frame with residual refinement

**Reference:** "Softmax Splatting for Video Frame Interpolation"
(Niklaus & Liu, CVPR 2020)

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `SoftSplat(NeuralNetworkArchitecture<>,SoftSplatOptions,IGradientBasedOptimizer<,Tensor<>,Tensor<>>)` | Creates a SoftSplat model in native training mode. |
| `SoftSplat(NeuralNetworkArchitecture<>,String,SoftSplatOptions)` | Creates a SoftSplat model in ONNX inference mode. |

## Methods

| Method | Summary |
|:-----|:--------|
| `Interpolate(Tensor<>,Tensor<>,Double)` | Interpolates between two frames using softmax splatting at timestep t. |

