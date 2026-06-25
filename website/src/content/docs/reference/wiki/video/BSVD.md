---
title: "BSVD<T>"
description: "BSVD bidirectional streaming video denoising with real-time buffers."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Video.Denoising`

BSVD bidirectional streaming video denoising with real-time buffers.

## For Beginners

BSVD (Blind Spot Video Denoising) removes noise from video without needing clean reference frames. It uses a blind-spot network that learns denoising patterns directly from noisy video data.

## How It Works

**References:**

- Paper: "BSVD: Bidirectional Streaming Video Denoising" (Qi et al., ACM MM 2022)

BSVD enables real-time video denoising through bidirectional streaming with efficient
buffer management. It processes video in forward and backward passes, maintaining compact
latent buffers for constant-memory operation regardless of video length.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `BSVD(NeuralNetworkArchitecture<>,BSVDOptions,IGradientBasedOptimizer<,Tensor<>,Tensor<>>)` | Creates a BSVD model for native training and inference. |
| `BSVD(NeuralNetworkArchitecture<>,String,BSVDOptions)` | Creates a BSVD model for ONNX inference. |

## Methods

| Method | Summary |
|:-----|:--------|
| `CreateNewInstance` |  |
| `Denoise(Tensor<>)` |  |
| `DeserializeNetworkSpecificData(BinaryReader)` |  |
| `Dispose(Boolean)` |  |
| `ForwardForTraining(Tensor<>)` | Same rationale as `Tensor{`: the tape-based `Tensor{` path runs `ForwardForTraining` on the raw input. |
| `GetModelMetadata` |  |
| `GetNamedLayerActivations(Tensor<>)` | Route the generic inspection path (used by `Tensor{` and test harnesses) through the same preprocessing that `Tensor{` applies. |
| `GetOptions` |  |
| `InitializeLayers` |  |
| `PostprocessOutput(Tensor<>)` |  |
| `PreprocessFrames(Tensor<>)` |  |
| `SerializeNetworkSpecificData(BinaryWriter)` |  |
| `Train(Tensor<>,Tensor<>)` |  |
| `UpdateParameters(Vector<>)` |  |

