---
title: "RPKNet<T>"
description: "RPKNet recurrent partial kernel network with separable large kernels for flow."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Video.Motion`

RPKNet recurrent partial kernel network with separable large kernels for flow.

## For Beginners

RPKNet (Recurrent Position-aware Kernel Network) uses position-aware convolution kernels that adapt to each pixel position for accurate optical flow estimation.

## How It Works

**References:**

- Paper: "RPKNet: Recurrent Partial Kernel Network for Efficient Optical Flow" (Morimitsu et al., AAAI 2024)

RPKNet uses recurrent partial kernel processing with separable large kernels for variable multi-scale feature extraction in optical flow.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `RPKNet` | Initializes a new instance with default architecture settings. |
| `RPKNet(NeuralNetworkArchitecture<>,Int32,Int32,RPKNetOptions)` | Creates a new RPKNet model for native training and inference. |

## Methods

| Method | Summary |
|:-----|:--------|
| `CreateNewInstance` |  |
| `DeserializeNetworkSpecificData(BinaryReader)` |  |
| `EstimateFlow(Tensor<>,Tensor<>)` |  |
| `GetModelMetadata` |  |
| `GetOptions` |  |
| `InitializeLayers` |  |
| `PostprocessOutput(Tensor<>)` |  |
| `PreprocessFrames(Tensor<>)` |  |
| `SerializeNetworkSpecificData(BinaryWriter)` |  |
| `Train(Tensor<>,Tensor<>)` |  |
| `UpdateParameters(Vector<>)` |  |

