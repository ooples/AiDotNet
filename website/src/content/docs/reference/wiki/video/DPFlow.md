---
title: "DPFlow<T>"
description: "DPFlow dual-pyramid framework combining image and feature pyramid advantages."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Video.Motion`

DPFlow dual-pyramid framework combining image and feature pyramid advantages.

## For Beginners

DPFlow (Dual-Path Flow) estimates optical flow using parallel spatial and temporal processing paths. The dual-path design captures both fine spatial detail and broad temporal motion patterns.

## How It Works

**References:**

- Paper: "DPFlow: Dual-Pyramid Optical Flow" (Morimitsu et al., CVPR 2025)

DPFlow combines the advantages of image pyramids (capturing large motions) and feature pyramids (rich semantics) in a unified dual-pyramid framework.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `DPFlow` | Initializes a new instance with default architecture settings. |
| `DPFlow(NeuralNetworkArchitecture<>,Int32,Int32,DPFlowOptions)` | Creates a new DPFlow model for native training and inference. |

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
| `TryGetArchitectureInputShape` | DPFlow's architecture stores the per-frame channel count, while the first model layer consumes the two-frame concatenation. |
| `UpdateParameters(Vector<>)` |  |

