---
title: "BiMVFIOptions"
description: "Configuration options for the BiMVFI bidirectional motion field model."
section: "API Reference"
---

`Options & Configuration` · `AiDotNet.Video.Options`

Configuration options for the BiMVFI bidirectional motion field model.

## For Beginners

When objects move at different speeds or occlude each other, a single
motion estimate fails. BiMVFI solves this by estimating motion from both directions (past
and future) and letting each pixel choose which direction gives a better result. Where an
object appears in one direction but not the other (occlusion), it knows to trust only the
visible direction.

## How It Works

BiMVFI (Seo et al., CVPR 2025) handles non-uniform motion with bidirectional fields:

- Bidirectional motion fields: estimates forward (0 to t) and backward (1 to t) motion

fields independently, each with its own confidence map, instead of a single symmetric flow

- Adaptive blending: per-pixel confidence weights learned from both motion fields determine

how to blend warped frames, handling occlusion regions where only one direction is valid

- Non-uniform motion modeling: dedicated occlusion reasoning module that detects regions with

non-uniform motion (e.g., independently moving objects) and applies motion-compensated
attention to those areas specifically

- Multi-scale architecture: 3-level feature pyramid with cross-scale feature propagation

for handling both small and large displacements

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `BiMVFIOptions` | Initializes a new instance with default values. |
| `BiMVFIOptions(BiMVFIOptions)` | Initializes a new instance by copying from another instance. |

## Properties

| Property | Summary |
|:-----|:--------|
| `ConfidenceThreshold` | Gets or sets the confidence threshold for adaptive blending. |
| `DropoutRate` | Gets or sets the dropout rate. |
| `LearningRate` | Gets or sets the learning rate. |
| `ModelPath` | Gets or sets the path to the ONNX model file. |
| `NumFeatures` | Gets or sets the number of feature channels. |
| `NumResBlocks` | Gets or sets the number of residual blocks in the motion estimator. |
| `NumScales` | Gets or sets the number of pyramid scales. |
| `OcclusionAwareBlending` | Gets or sets whether to use occlusion-aware blending. |
| `OnnxOptions` | Gets or sets the ONNX runtime options. |
| `Variant` | Gets or sets the model variant. |

