---
title: "RealViformerOptions"
description: "Configuration options for the RealViformer video informer for real-world video SR."
section: "API Reference"
---

`Options & Configuration` · `AiDotNet.Video.Options`

Configuration options for the RealViformer video informer for real-world video SR.

## For Beginners

RealViformer is designed for real-world video (phone recordings,
compressed streams), not just lab-quality test videos. It found that paying attention
to "which color channels matter" (channel attention) works better than "which spatial
locations matter" for handling the messy, complex degradations in real footage. It also
uses a trick from time-series forecasting (Informer) to efficiently handle longer videos.

## How It Works

RealViformer (Zhang and Yao, ECCV 2024) investigates attention for real-world VSR:

- Channel attention (CA): SE-style channel attention that recalibrates feature channels

based on global statistics, found more effective than spatial attention for real-world
degradations with complex noise patterns

- Temporal propagation: bidirectional recurrent feature propagation with channel

attention at each step for adaptive temporal fusion

- Informer-style sparse attention: ProbSparse self-attention that selects only the

top-k most informative queries, reducing quadratic complexity for longer sequences

- Real-world degradation handling: trained with second-order degradation modeling

(blur, noise, resize, JPEG) for practical video restoration

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `RealViformerOptions` | Initializes a new instance with default values. |
| `RealViformerOptions(RealViformerOptions)` | Initializes a new instance by copying from another instance. |

## Properties

| Property | Summary |
|:-----|:--------|
| `ChannelReductionRatio` | Gets or sets the channel attention reduction ratio. |
| `DropoutRate` | Gets or sets the dropout rate. |
| `LearningRate` | Gets or sets the learning rate. |
| `ModelPath` | Gets or sets the path to the ONNX model file. |
| `NumFeatures` | Gets or sets the number of feature channels. |
| `NumResBlocks` | Gets or sets the number of residual channel attention blocks. |
| `OnnxOptions` | Gets or sets the ONNX runtime options. |
| `ScaleFactor` | Gets or sets the spatial upscaling factor. |
| `SparseTopKFactor` | Gets or sets the ProbSparse attention top-k sampling factor. |
| `Variant` | Gets or sets the model variant. |

