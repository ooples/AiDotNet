---
title: "RealViformer<T>"
description: "RealViformer: investigating attention for real-world video super-resolution."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Video.Enhancement`

RealViformer: investigating attention for real-world video super-resolution.

## For Beginners

RealViformer is designed for real phone recordings and compressed
video streams, not just lab-quality test clips. It discovered that "which color channels
matter" (channel attention) works better than "which spatial locations matter" for the
messy, complex degradations in real footage. It also borrows an efficiency trick from
time-series forecasting to handle longer videos without running out of memory.

**Usage:**

## How It Works

RealViformer (Zhang and Yao, ECCV 2024) investigates attention for real-world VSR:

- Channel attention dominance: empirically shows that SE-style channel attention

outperforms spatial attention for real-world degradations with complex noise

- Bidirectional recurrent propagation: temporal feature propagation with channel

attention at each recurrent step for adaptive per-frame fusion weights

- ProbSparse attention: Informer-style sparse self-attention that selects only the

top-k most informative queries based on KL-divergence scoring, enabling efficient
processing of longer video sequences

- Real-world degradation training: second-order degradation model (blur, downsample,

noise, JPEG applied twice) for handling practical video artifacts

**Reference:** "RealViformer: Investigating Attention for Real-World Video
Super-Resolution" (Zhang and Yao, ECCV 2024)

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `RealViformer(NeuralNetworkArchitecture<>,RealViformerOptions,IGradientBasedOptimizer<,Tensor<>,Tensor<>>)` | Creates a RealViformer model in native training mode. |
| `RealViformer(NeuralNetworkArchitecture<>,String,RealViformerOptions)` | Creates a RealViformer model in ONNX inference mode. |

## Methods

| Method | Summary |
|:-----|:--------|
| `Upscale(Tensor<>)` |  |

