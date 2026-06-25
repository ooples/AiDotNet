---
title: "MemFlow<T>"
description: "MemFlow optical flow with memory for real-time historical motion aggregation."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Video.Motion`

MemFlow optical flow with memory for real-time historical motion aggregation.

## For Beginners

MemFlow uses memory-efficient transformers for optical flow estimation. It reduces memory consumption while maintaining high accuracy through a chunked attention mechanism.

## How It Works

**References:**

- Paper: "MemFlow: Optical Flow Estimation and Prediction with Memory" (Dong et al., CVPR 2024)

MemFlow augments flow estimation with an explicit memory module that aggregates historical motion information for improved temporal consistency.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `MemFlow` | Creates a new MemFlow model for native training and inference. |

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

