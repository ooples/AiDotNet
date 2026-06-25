---
title: "DUTOptions"
description: "Configuration options for the DUT (Deep Unsupervised Trajectory) video stabilization model."
section: "API Reference"
---

`Options & Configuration` · `AiDotNet.Video.Options`

Configuration options for the DUT (Deep Unsupervised Trajectory) video stabilization model.

## For Beginners

DUT options configure the Deep Unsupervised Trajectory video stabilizer.

## How It Works

**References:**

- Paper: "DUT: Learning Video Stabilization by Simply Watching Unstable Videos" (Xu et al., ICCV 2022)

DUT learns stabilization in an unsupervised manner by watching unstable videos,
predicting per-pixel flow fields for warping without requiring paired stable/unstable data.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `DUTOptions` | Initializes a new instance with default values. |
| `DUTOptions(DUTOptions)` | Initializes a new instance by copying from another instance. |

## Properties

| Property | Summary |
|:-----|:--------|
| `DropoutRate` | Dropout rate for regularization. |
| `LearningRate` | Learning rate for training. |
| `ModelPath` | Path to the ONNX model file for inference mode. |
| `NumFeatures` | Number of base feature channels in the flow estimation network. |
| `NumPyramidLevels` | Number of pyramid levels for coarse-to-fine flow estimation. |
| `NumResBlocks` | Number of residual blocks per pyramid level. |
| `OnnxOptions` | ONNX runtime options for inference mode. |
| `TemporalLossWeight` | Weight for the temporal consistency loss during training. |
| `TemporalWindowSize` | Temporal window size for multi-frame stabilization. |
| `Variant` | Model variant controlling capacity and speed trade-off. |

