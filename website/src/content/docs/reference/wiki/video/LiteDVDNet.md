---
title: "LiteDVDNet<T>"
description: "LiteDVDNet lightweight deep video denoising with depthwise separable convolutions."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Video.Denoising`

LiteDVDNet lightweight deep video denoising with depthwise separable convolutions.

## For Beginners

LiteDVDNet is a lightweight video denoiser designed for real-time performance. It achieves good denoising quality with significantly fewer parameters than full-scale models like DVDNet.

## How It Works

**References:**

- Paper: "LiteDVDNet: A Lightweight Deep Video Denoising Network" (2020)

LiteDVDNet is an efficient two-stage denoiser that first processes frames independently
then fuses temporal information, using depthwise separable convolutions for 8-10x
parameter reduction while maintaining quality.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `LiteDVDNet(NeuralNetworkArchitecture<>,LiteDVDNetOptions,IGradientBasedOptimizer<,Tensor<>,Tensor<>>)` | Creates a LiteDVDNet model for native training and inference. |
| `LiteDVDNet(NeuralNetworkArchitecture<>,String,LiteDVDNetOptions)` | Creates a LiteDVDNet model for ONNX inference. |

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

