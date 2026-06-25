---
title: "DKM<T>"
description: "DKM dense kernelized feature matching for geometry estimation."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Video.Motion`

DKM dense kernelized feature matching for geometry estimation.

## For Beginners

DKM (Dense Kernelized Matching) estimates dense correspondences between image pairs using learned kernels. It produces accurate pixel-level matching for optical flow and stereo vision.

## How It Works

**References:**

- Paper: "DKM: Dense Kernelized Feature Matching for Geometry Estimation" (Edstedt et al., CVPR 2023)

DKM uses dense kernelized matching to establish pixel-level correspondences between images for accurate geometry estimation.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `DKM` | Creates a new DKM model for native training and inference. |

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

