---
title: "NeuFlowV2<T>"
description: "NeuFlow v2 high-efficiency optical flow on edge devices."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Video.Motion`

NeuFlow v2 high-efficiency optical flow on edge devices.

## For Beginners

NeuFlow V2 is a fast, lightweight optical flow estimator designed for real-time applications. It achieves good accuracy with significantly reduced computation compared to transformer-based methods.

## How It Works

**References:**

- Paper: "NeuFlow v2: High-Efficiency Optical Flow Estimation on Edge Devices" (Zhang et al., 2024)

NeuFlow v2 achieves high-efficiency optical flow estimation suitable for edge devices through a lightweight backbone and optimized inference.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `NeuFlowV2` | Creates a new NeuFlowV2 model for native training and inference. |

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

