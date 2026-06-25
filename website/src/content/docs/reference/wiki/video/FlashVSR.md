---
title: "FlashVSR<T>"
description: "FlashVSR: one-step diffusion streaming video super-resolution at real-time speed."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Video.Enhancement`

FlashVSR: one-step diffusion streaming video super-resolution at real-time speed.

## For Beginners

FlashVSR makes low-resolution video sharper in real time.
Most AI upscalers are too slow for live video because they need many processing steps.
FlashVSR solves this by learning to do it in just one step, making it fast enough
for streaming at ~17 frames per second while maintaining high quality.

**Usage:**

## How It Works

FlashVSR (Zhuang et al., 2025) achieves ~17 FPS streaming 4x video super-resolution
through a one-step diffusion framework with three key innovations:

- Locality-Constrained Sparse Attention (LCSA) for efficient spatial feature extraction
- A tiny conditional decoder that produces HR output in a single denoising step
- Flow-guided deformable alignment for temporal fusion across frames

The model is trained via knowledge distillation from a multi-step diffusion teacher,
compressing 20+ denoising steps into a single forward pass.

**Reference:** "FlashVSR: Efficient Real-Time Video Super-Resolution via One-Step Diffusion"
(Zhuang et al., 2025)

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `FlashVSR` | Creates a FlashVSR model with default architecture (paper-default 128x128x3 input, scale x4) in native training mode. |
| `FlashVSR(NeuralNetworkArchitecture<>,FlashVSROptions,IGradientBasedOptimizer<,Tensor<>,Tensor<>>)` | Creates a FlashVSR model in native training mode. |
| `FlashVSR(NeuralNetworkArchitecture<>,String,FlashVSROptions)` | Creates a FlashVSR model in ONNX inference mode. |

## Methods

| Method | Summary |
|:-----|:--------|
| `Upscale(Tensor<>)` |  |

