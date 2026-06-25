---
title: "RoMa<T>"
description: "RoMa robust dense feature matching with DINOv2 foundation for pixel-dense warps."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Video.Motion`

RoMa robust dense feature matching with DINOv2 foundation for pixel-dense warps.

## For Beginners

RoMa (Robust Dense Matching) estimates robust pixel correspondences between images. It handles challenging cases like occlusions, textureless regions, and large displacements.

## How It Works

**References:**

- Paper: "RoMa: Robust Dense Feature Matching" (Edstedt et al., CVPR 2024)

RoMa achieves robust dense feature matching using DINOv2 foundation features, producing pixel-dense correspondence maps for geometry estimation.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `RoMa` | Initializes a new instance with default architecture settings. |
| `RoMa(NeuralNetworkArchitecture<>,Int32,Int32,RoMaOptions)` | Creates a new RoMa model for native training and inference. |

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

