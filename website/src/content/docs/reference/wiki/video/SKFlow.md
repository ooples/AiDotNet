---
title: "SKFlow<T>"
description: "SKFlow selective kernel attention for efficient optical flow estimation."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Video.Motion`

SKFlow selective kernel attention for efficient optical flow estimation.

## For Beginners

SKFlow uses selective kernels that adapt their receptive field size based on local motion patterns. Different kernel sizes handle different motion scales automatically.

## How It Works

**References:**

- Paper: "SKFlow: Learning Optical Flow with Super Kernels" (Sun et al., 2022)

SKFlow uses selective kernel attention mechanisms to efficiently capture multi-scale motion information for optical flow estimation.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `SKFlow` | Creates a new SKFlow model for native training and inference. |

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

