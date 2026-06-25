---
title: "FlowFormerPlusPlus<T>"
description: "FlowFormer++ masked cost volume autoencoding with tile-based high-resolution flow."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Video.Motion`

FlowFormer++ masked cost volume autoencoding with tile-based high-resolution flow.

## For Beginners

FlowFormer++ improves on FlowFormer with enhanced cost volume processing and better handling of occlusions. It achieves state-of-the-art accuracy on optical flow benchmarks.

## How It Works

**References:**

- Paper: "FlowFormer++: Masked Cost Volume Autoencoding for Pretraining Optical Flow Estimation" (Shi et al., 2023)

FlowFormer++ extends FlowFormer with masked cost volume autoencoding pretraining and tile-based processing for high-resolution optical flow.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `FlowFormerPlusPlus` | Creates a new FlowFormerPlusPlus model for native training and inference. |

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

