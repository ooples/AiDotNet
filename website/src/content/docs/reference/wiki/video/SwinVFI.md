---
title: "SwinVFI<T>"
description: "SwinVFI: Swin Transformer-based video frame interpolation."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Video.FrameInterpolation`

SwinVFI: Swin Transformer-based video frame interpolation.

## For Beginners

SwinVFI uses the Swin Transformer to look at both input frames
simultaneously and figure out what goes between them, without needing to estimate motion.
The "shifted window" approach makes it efficient for full-resolution video frames.

**Usage:**

## How It Works

SwinVFI (2022) applies Swin Transformer architecture to frame interpolation:

- Swin Transformer encoder: uses shifted-window self-attention with linear complexity for

encoding input frame pairs at high resolution

- Cross-frame window attention: extends shifted-window attention to cross-attend between

features from both input frames, capturing inter-frame correspondences

- Hierarchical feature pyramid: multi-scale feature extraction with Swin blocks at each level
- Flow-free synthesis: directly synthesizes the intermediate frame from cross-attended

features without explicit optical flow estimation

**Reference:** "SwinVFI: Swin Transformer-based Video Frame Interpolation" (2022)

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `SwinVFI(NeuralNetworkArchitecture<>,String,SwinVFIOptions)` | Creates a SwinVFI model in ONNX inference mode. |
| `SwinVFI(NeuralNetworkArchitecture<>,SwinVFIOptions,IGradientBasedOptimizer<,Tensor<>,Tensor<>>)` | Creates a SwinVFI model in native training mode. |

## Methods

| Method | Summary |
|:-----|:--------|
| `Interpolate(Tensor<>,Tensor<>,Double)` |  |

