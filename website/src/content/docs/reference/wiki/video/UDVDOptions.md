---
title: "UDVDOptions"
description: "Configuration options for UDVD unidirectional deep video denoising."
section: "API Reference"
---

`Options & Configuration` · `AiDotNet.Video.Options`

Configuration options for UDVD unidirectional deep video denoising.

## For Beginners

UDVD removes noise from video without needing clean reference
footage for training. It learns to denoise by recognizing that noise is random (different
each frame) while real content is consistent. This makes it especially useful for
real-world noisy video where you don't have a clean version to compare against.

## How It Works

UDVD (Sheth et al., CVPR 2021) performs blind video denoising without paired training data:

- Blind denoising: requires only noisy video for training (no clean ground truth),

using a self-supervised loss that exploits temporal redundancy

- Unidirectional: processes frames in a single forward pass using only past frames,

enabling real-time streaming operation

- Multi-frame fusion: combines features from multiple past frames with learned weights

that adapt to content and noise characteristics

- Noise-adaptive: handles varying and unknown noise levels including real camera noise

(not just synthetic Gaussian), making it practical for real-world footage

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `UDVDOptions` | Initializes a new instance with default values. |
| `UDVDOptions(UDVDOptions)` | Initializes a new instance by copying from another instance. |

## Properties

| Property | Summary |
|:-----|:--------|
| `DropoutRate` | Gets or sets the dropout rate. |
| `LearningRate` | Gets or sets the learning rate. |
| `ModelPath` | Gets or sets the path to the ONNX model file. |
| `NumFeatures` | Gets or sets the number of feature channels. |
| `NumLevels` | Gets or sets the number of U-Net levels. |
| `NumResBlocks` | Gets or sets the number of residual blocks per level. |
| `OnnxOptions` | Gets or sets the ONNX runtime options. |
| `TemporalBufferSize` | Gets or sets the temporal buffer size (past frames to use). |
| `Variant` | Gets or sets the model variant. |

