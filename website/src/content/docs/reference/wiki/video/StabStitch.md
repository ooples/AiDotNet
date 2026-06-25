---
title: "StabStitch<T>"
description: "StabStitch joint video stabilization and stitching for panoramic output."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Video.Stabilization`

StabStitch joint video stabilization and stitching for panoramic output.

## For Beginners

StabStitch jointly stabilizes and stitches video for panoramic applications. It produces smooth, wide-angle video from multiple camera feeds or a moving camera.

## How It Works

**References:**

- Paper: "StabStitch: Simultaneous Video Stabilization and Stitching" (2023)

StabStitch jointly optimizes video stabilization and stitching from moving cameras,
using thin-plate-spline mesh warping to simultaneously remove camera shake and produce
a seamless panoramic output without separate stabilization and stitching stages.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `StabStitch(NeuralNetworkArchitecture<>,StabStitchOptions,IGradientBasedOptimizer<,Tensor<>,Tensor<>>)` | Creates a StabStitch model for native training and inference. |
| `StabStitch(NeuralNetworkArchitecture<>,String,StabStitchOptions)` | Creates a StabStitch model for ONNX inference. |

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

