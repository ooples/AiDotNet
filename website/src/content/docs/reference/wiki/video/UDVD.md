---
title: "UDVD<T>"
description: "UDVD unidirectional deep video denoising for blind self-supervised denoising."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Video.Denoising`

UDVD unidirectional deep video denoising for blind self-supervised denoising.

## For Beginners

UDVD (Unidirectional Video Denoising) processes video in a single temporal direction for causal denoising. This makes it suitable for streaming applications where future frames are not available.

## How It Works

**References:**

- Paper: "Unsupervised Deep Video Denoising" (Sheth et al., CVPR 2021)

UDVD performs blind video denoising without paired training data. In the original paper,
training uses a self-supervised loss that exploits temporal redundancy. The native Train
method uses a supervised approach with paired clean/noisy data for simplicity; the full
self-supervised training pipeline is available through the ONNX model. It processes frames
unidirectionally using only past frames, enabling real-time streaming operation.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `UDVD(NeuralNetworkArchitecture<>,String,UDVDOptions)` | Creates a UDVD model for ONNX inference. |
| `UDVD(NeuralNetworkArchitecture<>,UDVDOptions,IGradientBasedOptimizer<,Tensor<>,Tensor<>>)` | Creates a UDVD model for native training and inference. |

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

