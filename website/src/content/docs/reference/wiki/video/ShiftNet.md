---
title: "ShiftNet<T>"
description: "ShiftNet channel-shifting video denoiser using zero-cost temporal feature exchange."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Video.Denoising`

ShiftNet channel-shifting video denoiser using zero-cost temporal feature exchange.

## For Beginners

ShiftNet uses efficient shift operations instead of expensive 3D convolutions for video denoising. By shifting feature maps along the temporal dimension, it captures motion at minimal computational cost.

## How It Works

**References:**

- Paper: "An Efficient Recurrent Architecture for Video Denoising via Temporal Shift" (Maggioni et al., 2021)

ShiftNet shifts feature channels along the temporal dimension without explicit alignment,
using a U-Net backbone where each conv block incorporates channel shifting for temporal
awareness, avoiding expensive optical flow or attention.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `ShiftNet(NeuralNetworkArchitecture<>,ShiftNetOptions,IGradientBasedOptimizer<,Tensor<>,Tensor<>>)` | Creates a ShiftNet model for native training and inference. |
| `ShiftNet(NeuralNetworkArchitecture<>,String,ShiftNetOptions)` | Creates a ShiftNet model for ONNX inference. |

## Methods

| Method | Summary |
|:-----|:--------|
| `CreateNewInstance` |  |
| `Denoise(Tensor<>)` |  |
| `DeserializeNetworkSpecificData(BinaryReader)` |  |
| `Dispose(Boolean)` |  |
| `GetModelMetadata` |  |
| `GetOptions` |  |
| `InitializeLayers` |  |
| `PostprocessOutput(Tensor<>)` |  |
| `PreprocessFrames(Tensor<>)` |  |
| `SerializeNetworkSpecificData(BinaryWriter)` |  |
| `Train(Tensor<>,Tensor<>)` |  |
| `UpdateParameters(Vector<>)` |  |

