---
title: "FuSta<T>"
description: "FuSta hybrid full-frame video stabilization with warping and outpainting."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Video.Stabilization`

FuSta hybrid full-frame video stabilization with warping and outpainting.

## For Beginners

FuSta (Fusion Stabilization) stabilizes video by fusing multiple stabilization strategies including trajectory smoothing and homography warping for robust results.

## How It Works

**References:**

- Paper: "FuSta: Hybrid Approach for Full-frame Video Stabilization" (Liu et al., 2021)

FuSta achieves full-frame stabilization through a two-stage approach: first warping frames
using optical-flow-based motion compensation, then using a neural outpainting network
to fill missing border regions, avoiding the field-of-view loss of traditional cropping.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `FuSta(NeuralNetworkArchitecture<>,FuStaOptions,IGradientBasedOptimizer<,Tensor<>,Tensor<>>)` | Creates a FuSta model for native training and inference. |
| `FuSta(NeuralNetworkArchitecture<>,String,FuStaOptions)` | Creates a FuSta model for ONNX inference. |

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

