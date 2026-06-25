---
title: "TDPNet<T>"
description: "TDPNet: temporal difference prediction network for video frame interpolation."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Video.FrameInterpolation`

TDPNet: temporal difference prediction network for video frame interpolation.

## For Beginners

TDPNet starts with a simple average of the two frames and predicts
what needs to change, like correcting a rough draft rather than writing from blank.

**Usage:**

## How It Works

TDPNet (2024) predicts temporal differences for efficient interpolation:

- Temporal difference prediction: predicts only the residual between the intermediate frame

and a linear blend, focusing network capacity on the non-trivial parts

- Difference-aware attention: attends specifically to regions where the temporal difference

is large (motion boundaries, occlusions)

- Coarse-to-fine refinement: multi-scale architecture where coarse differences capture global

corrections and fine differences add sharp texture details

- Lightweight backbone: predicting residuals is easier than full frames, enabling a lighter

network with comparable quality

**Reference:** "TDPNet: Temporal Difference Prediction Network for Video Frame Interpolation" (2024)

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `TDPNet(NeuralNetworkArchitecture<>,String,TDPNetOptions)` | Creates a TDPNet model in ONNX inference mode. |
| `TDPNet(NeuralNetworkArchitecture<>,TDPNetOptions,IGradientBasedOptimizer<,Tensor<>,Tensor<>>)` | Creates a TDPNet model in native training mode. |

## Methods

| Method | Summary |
|:-----|:--------|
| `Interpolate(Tensor<>,Tensor<>,Double)` |  |

