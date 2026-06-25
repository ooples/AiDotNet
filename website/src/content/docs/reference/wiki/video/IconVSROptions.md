---
title: "IconVSROptions"
description: "Configuration options for the IconVSR information-aggregation video super-resolution model."
section: "API Reference"
---

`Options & Configuration` · `AiDotNet.Video.Options`

Configuration options for the IconVSR information-aggregation video super-resolution model.

## For Beginners

While BasicVSR processes frames one by one, IconVSR picks
important "keyframes" and uses them as extra reference points. This helps
especially for long video sequences where errors can build up over time.

## How It Works

IconVSR (Chan et al., CVPR 2021) extends BasicVSR with two key modules:

- Information-Aggregation Module: extracts features from sparsely-selected keyframes

and uses them to refine propagation features via a cross-attention mechanism

- Coupled Propagation: forward and backward branches exchange information to reduce

error accumulation in long sequences

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `IconVSROptions` | Initializes a new instance with default values. |
| `IconVSROptions(IconVSROptions)` | Initializes a new instance by copying from another instance. |

## Properties

| Property | Summary |
|:-----|:--------|
| `DropoutRate` | Gets or sets the dropout rate. |
| `KeyframeStride` | Gets or sets the stride for selecting keyframes from the sequence. |
| `LearningRate` | Gets or sets the learning rate. |
| `ModelPath` | Gets or sets the path to the ONNX model file. |
| `NumEdemaBlocks` | Gets or sets the number of EDVR-style deformable alignment blocks in the information-aggregation module. |
| `NumFeatures` | Gets or sets the number of feature channels. |
| `NumFrames` | Gets or sets the number of input frames. |
| `NumResBlocks` | Gets or sets the number of residual blocks per propagation direction. |
| `OnnxOptions` | Gets or sets the ONNX runtime options. |
| `ScaleFactor` | Gets or sets the spatial upscaling factor. |
| `Variant` | Gets or sets the model variant. |
| `WarmupSteps` | Gets or sets the warmup steps for the learning rate schedule. |

