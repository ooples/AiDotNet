---
title: "ThreeDMF<T>"
description: "3DMF 3D motion field video stabilization with depth-aware camera compensation."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Video.Stabilization`

3DMF 3D motion field video stabilization with depth-aware camera compensation.

## For Beginners

3DMF (3D Motion Field) stabilizes video by estimating 3D camera motion and removing unwanted shake while preserving intentional camera movements like pans and tilts.

## How It Works

**References:**

- Paper: "3D Video Stabilization with Depth Estimation by CNN-based Optimization" (Lee & Lee, CVPR 2021)

3DMF estimates depth and 3D camera motion to perform stabilization in 3D space,
jointly predicting per-pixel depth maps and 6-DOF camera poses to compute
depth-aware warping that correctly handles parallax and depth-dependent motion.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `ThreeDMF(NeuralNetworkArchitecture<>,String,ThreeDMFOptions)` | Creates a ThreeDMF model for ONNX inference. |
| `ThreeDMF(NeuralNetworkArchitecture<>,ThreeDMFOptions,IGradientBasedOptimizer<,Tensor<>,Tensor<>>)` | Creates a ThreeDMF model for native training and inference. |

## Methods

| Method | Summary |
|:-----|:--------|
| `CreateNewInstance` |  |
| `DeserializeNetworkSpecificData(BinaryReader)` |  |
| `Dispose(Boolean)` |  |
| `GetModelMetadata` |  |
| `GetOptions` |  |
| `InitializeLayers` |  |
| `PostprocessOutput(Tensor<>)` |  |
| `PreprocessFrames(Tensor<>)` |  |
| `SerializeNetworkSpecificData(BinaryWriter)` |  |
| `Stabilize(Tensor<>)` |  |
| `Train(Tensor<>,Tensor<>)` |  |
| `UpdateParameters(Vector<>)` |  |

