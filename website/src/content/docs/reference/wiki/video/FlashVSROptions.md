---
title: "FlashVSROptions"
description: "Configuration options for the FlashVSR real-time streaming video super-resolution model."
section: "API Reference"
---

`Options & Configuration` · `AiDotNet.Video.Options`

Configuration options for the FlashVSR real-time streaming video super-resolution model.

## For Beginners

FlashVSR is a video upscaler that makes low-resolution video look sharper
in real time. Most diffusion-based methods need 20-50 steps (very slow), but FlashVSR does it
in just one step by using a distilled model, making it fast enough for live streaming.

## How It Works

FlashVSR (Zhuang et al., 2025) achieves real-time 4x video super-resolution (~17 FPS)
through a one-step diffusion framework with three key innovations:

- Locality-Constrained Sparse Attention (LCSA): limits attention to local windows for efficiency
- Tiny Conditional Decoder: lightweight decoder that generates HR output in a single diffusion step
- Flow-guided temporal alignment: deformable convolution guided by optical flow for multi-frame fusion

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `FlashVSROptions` | Initializes a new instance with default values. |
| `FlashVSROptions(FlashVSROptions)` | Initializes a new instance by copying from another instance. |

## Properties

| Property | Summary |
|:-----|:--------|
| `DistillationWeight` | Gets or sets the distillation loss weight (teacher-student). |
| `DropoutRate` | Gets or sets the dropout rate. |
| `FeedForwardExpansion` | Gets or sets the feed-forward expansion factor in LCSA blocks. |
| `LearningRate` | Gets or sets the learning rate. |
| `ModelPath` | Gets or sets the path to the ONNX model file. |
| `NumDecoderBlocks` | Gets or sets the number of residual blocks in the conditional decoder. |
| `NumFeatures` | Gets or sets the number of feature channels in the encoder and decoder. |
| `NumHeads` | Gets or sets the number of attention heads in LCSA. |
| `NumInputFrames` | Gets or sets the number of input frames for temporal alignment. |
| `NumLCSABlocks` | Gets or sets the number of Locality-Constrained Sparse Attention blocks. |
| `OnnxOptions` | Gets or sets the ONNX runtime options. |
| `ScaleFactor` | Gets or sets the spatial upscaling factor. |
| `Variant` | Gets or sets the model variant. |
| `WarmupSteps` | Gets or sets the number of warmup steps for the learning rate schedule. |
| `WindowSize` | Gets or sets the local window size for LCSA attention. |

