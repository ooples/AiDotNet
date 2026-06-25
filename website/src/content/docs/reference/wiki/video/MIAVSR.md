---
title: "MIAVSR<T>"
description: "MIA-VSR: masked inter and intra-frame attention for efficient video super-resolution."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Video.Enhancement`

MIA-VSR: masked inter and intra-frame attention for efficient video super-resolution.

## For Beginners

MIA-VSR makes video super-resolution faster by being selective.
Instead of comparing every pixel with every other pixel in neighboring frames (which is
very slow), it uses "masks" to focus only on the most important parts. It has two types:
inter-frame masks find the best matching regions across time, and intra-frame masks
enhance spatial details within each frame.

**Usage:**

## How It Works

MIA-VSR (Zhou et al., CVPR 2024) uses masked attention for efficient temporal modeling:

- Masked inter-frame attention: temporal attention across frames with sparse masking,

attending only to the most relevant spatial locations in neighboring frames rather
than all positions, reducing quadratic complexity

- Masked intra-frame attention: spatial attention within each frame with local window

masking that progressively decreases through layers (coarse to fine)

- BasicVSR++ backbone: bidirectional recurrent propagation with masked attention

replacing deformable alignment for improved quality and efficiency

- Progressive masking schedule: early layers use aggressive masking for speed,

later layers use less masking for quality

**Reference:** "MIA-VSR: Masked Inter and Intra-Frame Attention for Video
Super-Resolution" (Zhou et al., CVPR 2024)

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `MIAVSR(NeuralNetworkArchitecture<>,MIAVSROptions,IGradientBasedOptimizer<,Tensor<>,Tensor<>>)` | Creates a MIA-VSR model in native training mode. |
| `MIAVSR(NeuralNetworkArchitecture<>,String,MIAVSROptions)` | Creates a MIA-VSR model in ONNX inference mode. |

## Methods

| Method | Summary |
|:-----|:--------|
| `Upscale(Tensor<>)` |  |

