---
title: "VideoDenoisingBase<T>"
description: "Base class for video denoising models that remove noise from video sequences."
section: "API Reference"
---

`Base Classes` · `AiDotNet.Video`

Base class for video denoising models that remove noise from video sequences.

## For Beginners

Video denoising cleans up grainy or noisy video footage. This is
common in low-light video, security camera footage, or video shot at high ISO settings.
The model learns to distinguish real detail from random noise, removing the noise while
keeping important details sharp. Using multiple frames helps because noise is random
(different in each frame) while real content is consistent across frames.

## How It Works

Video denoising removes noise and grain from video while preserving detail and temporal
consistency. This base class provides:

- Noise level (sigma) handling for both blind and non-blind denoising
- Noise estimation utilities
- Temporal buffer management for multi-frame denoising

Derived classes implement specific architectures like FastDVDNet, BSVD, FloRNN, etc.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `VideoDenoisingBase(NeuralNetworkArchitecture<>,ILossFunction<>,Double)` | Initializes a new instance of the VideoDenoisingBase class. |

## Properties

| Property | Summary |
|:-----|:--------|
| `IsBlindDenoising` | Gets whether this model performs blind denoising (estimates noise level automatically). |
| `NoiseSigma` | Gets or sets the noise sigma level for non-blind denoising. |
| `TemporalRadius` | Gets the temporal radius (number of frames before and after used for context). |

## Methods

| Method | Summary |
|:-----|:--------|
| `Denoise(Tensor<>)` | Denoises a sequence of video frames. |
| `EstimateNoiseLevel(Tensor<>)` | Estimates the noise level (sigma) from noisy input frames. |
| `PredictCore(Tensor<>)` |  |

