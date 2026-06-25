---
title: "FloRNN<T>"
description: "FloRNN optical-flow-guided recurrent video denoising."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Video.Denoising`

FloRNN optical-flow-guided recurrent video denoising.

## For Beginners

FloRNN (Flow-guided Recurrent Neural Network) denoises video frames by using optical flow to align neighboring frames before applying recurrent processing. This flow-guided approach preserves temporal consistency.

## How It Works

**References:**

- Paper: "Flowing Recurrent Network for Video Denoising" (Li et al., AAAI 2022)

FloRNN uses optical flow to guide recurrent denoising, warping previous hidden states
for temporal alignment before feeding them to ConvLSTM/ConvGRU units, with occlusion-aware
gating to suppress unreliable aligned features.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `FloRNN(NeuralNetworkArchitecture<>,FloRNNOptions,IGradientBasedOptimizer<,Tensor<>,Tensor<>>)` | Creates a FloRNN model for native training and inference. |
| `FloRNN(NeuralNetworkArchitecture<>,String,FloRNNOptions)` | Creates a FloRNN model for ONNX inference. |

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

