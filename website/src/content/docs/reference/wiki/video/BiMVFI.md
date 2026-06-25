---
title: "BiMVFI<T>"
description: "BiMVFI: bidirectional motion field-based video frame interpolation."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Video.FrameInterpolation`

BiMVFI: bidirectional motion field-based video frame interpolation.

## For Beginners

When objects move at different speeds or occlude each other, a single
motion estimate fails. BiMVFI solves this by estimating motion from both directions (past
and future) and letting each pixel choose which direction gives a better result. Where an
object appears in one direction but not the other, it trusts the visible direction.

**Usage:**

## How It Works

BiMVFI (Seo et al., CVPR 2025) handles non-uniform motion with bidirectional fields:

- Bidirectional motion fields: estimates forward (0 to t) and backward (1 to t) motion

fields independently, each with its own confidence map, instead of a single symmetric flow

- Adaptive blending: per-pixel confidence weights learned from both motion fields determine

how to blend warped frames, gracefully handling occlusion regions where only one direction
provides valid information

- Non-uniform motion modeling: dedicated occlusion reasoning module detects regions with

non-uniform motion (e.g., independently moving objects) and applies motion-compensated
attention to those areas specifically

- Multi-scale architecture: 3-level feature pyramid with cross-scale feature propagation

for handling both small sub-pixel motions and large inter-frame displacements

**Reference:** "BiMVFI: Bidirectional Motion Field-Based Video Frame Interpolation"
(Seo et al., CVPR 2025)

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `BiMVFI(NeuralNetworkArchitecture<>,BiMVFIOptions,IGradientBasedOptimizer<,Tensor<>,Tensor<>>)` | Creates a BiMVFI model in native training mode. |
| `BiMVFI(NeuralNetworkArchitecture<>,String,BiMVFIOptions)` | Creates a BiMVFI model in ONNX inference mode. |

## Methods

| Method | Summary |
|:-----|:--------|
| `Interpolate(Tensor<>,Tensor<>,Double)` |  |

