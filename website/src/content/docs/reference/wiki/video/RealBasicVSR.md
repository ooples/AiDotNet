---
title: "RealBasicVSR<T>"
description: "RealBasicVSR: real-world video super-resolution with stochastic degradation simulation."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Video.Enhancement`

RealBasicVSR: real-world video super-resolution with stochastic degradation simulation.

## For Beginners

Real videos from phones or old cameras have noise, blur, and
compression artifacts. RealBasicVSR first "cleans" each frame to remove these issues,
then upscales the cleaned frames. This prevents noise from spreading through the
video during the upscaling process.

**Usage:**

## How It Works

RealBasicVSR (Chan et al., CVPR 2022) addresses real-world video SR through:

- Stochastic degradation scheme: randomly applies blur, noise, resize, and JPEG/H.264

compression during training to simulate diverse real-world quality issues

- Pre-cleaning module: a 20-block residual network that removes noise and artifacts

from each frame BEFORE it enters the recurrent propagation, preventing degradation
from spreading across the temporal dimension

- BasicVSR backbone: bidirectional recurrent propagation with flow-based alignment

The pre-cleaning module is the key innovation: without it, noise in one frame
propagates to all subsequent frames through the recurrent connections.

**Reference:** "Investigating Tradeoffs in Real-World Video Super-Resolution"
(Chan et al., CVPR 2022)

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `RealBasicVSR(NeuralNetworkArchitecture<>,RealBasicVSROptions,IGradientBasedOptimizer<,Tensor<>,Tensor<>>)` | Creates a RealBasicVSR model in native training mode. |
| `RealBasicVSR(NeuralNetworkArchitecture<>,String,RealBasicVSROptions)` | Creates a RealBasicVSR model in ONNX inference mode. |

## Methods

| Method | Summary |
|:-----|:--------|
| `Upscale(Tensor<>)` |  |

