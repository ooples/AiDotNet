---
title: "STMFNet<T>"
description: "STMFNet: spatio-temporal multi-flow network for video frame interpolation."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Video.FrameInterpolation`

STMFNet: spatio-temporal multi-flow network for video frame interpolation.

## For Beginners

STMFNet makes multiple motion guesses (flows) and picks the best one
for each part of the image. This handles tricky areas like object boundaries much better
than methods that make only one motion guess.

**Usage:**

## How It Works

STMFNet (2022) uses multiple optical flows in spatio-temporal space:

- Multi-flow estimation: estimates multiple (typically 4) optical flow fields, each capturing

different motion hypotheses for ambiguous regions

- Spatio-temporal feature volume: constructs a 4D feature volume from input frames and all

estimated flow fields, capturing the full motion context

- Flow selection network: selects the best flow hypothesis for each pixel by comparing

warped features from each flow field

- Residual refinement: after flow-based warping, corrects remaining artifacts using the

multi-flow feature volume as context

**Reference:** "STMFNet: Spatio-Temporal Multi-Flow Network for Video Frame Interpolation" (2022)

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `STMFNet(NeuralNetworkArchitecture<>,STMFNetOptions,IGradientBasedOptimizer<,Tensor<>,Tensor<>>)` | Creates an STMFNet model in native training mode. |
| `STMFNet(NeuralNetworkArchitecture<>,String,STMFNetOptions)` | Creates an STMFNet model in ONNX inference mode. |

## Methods

| Method | Summary |
|:-----|:--------|
| `Interpolate(Tensor<>,Tensor<>,Double)` |  |

