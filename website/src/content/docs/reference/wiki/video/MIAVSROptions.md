---
title: "MIAVSROptions"
description: "Configuration options for the MIA-VSR masked inter and intra-frame attention model."
section: "API Reference"
---

`Options & Configuration` · `AiDotNet.Video.Options`

Configuration options for the MIA-VSR masked inter and intra-frame attention model.

## For Beginners

MIA-VSR makes video super-resolution faster by being selective
about what it pays attention to. Instead of looking at every pixel in every frame
(which is slow), it uses "masks" to focus only on the most important parts. It looks
between frames (inter) to track moving objects and within frames (intra) to enhance
spatial details.

## How It Works

MIA-VSR (Zhou et al., CVPR 2024) uses masked attention for efficient video SR:

- Masked inter-frame attention: temporal attention across frames with sparse masking,

attending only to the most relevant spatial locations in neighboring frames

- Masked intra-frame attention: spatial attention within each frame with local window

masking for computational efficiency

- Progressive masking: the masking ratio decreases through layers, from coarse to fine
- Built on BasicVSR++ backbone with attention replacing deformable convolution

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `MIAVSROptions` | Initializes a new instance with default values. |
| `MIAVSROptions(MIAVSROptions)` | Initializes a new instance by copying from another instance. |

## Properties

| Property | Summary |
|:-----|:--------|
| `DropoutRate` | Gets or sets the dropout rate. |
| `InterMaskRatio` | Gets or sets the masking ratio for inter-frame attention (0.0-1.0). |
| `IntraMaskRatio` | Gets or sets the masking ratio for intra-frame attention (0.0-1.0). |
| `LearningRate` | Gets or sets the learning rate. |
| `ModelPath` | Gets or sets the path to the ONNX model file. |
| `NumFeatures` | Gets or sets the number of feature channels. |
| `NumHeads` | Gets or sets the number of attention heads for inter/intra-frame attention. |
| `NumResBlocks` | Gets or sets the number of residual blocks in each propagation branch. |
| `OnnxOptions` | Gets or sets the ONNX runtime options. |
| `ScaleFactor` | Gets or sets the spatial upscaling factor. |
| `Variant` | Gets or sets the model variant. |
| `WindowSize` | Gets or sets the window size for masked intra-frame attention. |

