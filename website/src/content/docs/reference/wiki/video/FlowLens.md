---
title: "FlowLens<T>"
description: "FlowLens optical-flow-guided video inpainting with flow completion."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Video.Inpainting`

FlowLens optical-flow-guided video inpainting with flow completion.

## For Beginners

FlowLens performs video inpainting by using optical flow as a lens to guide content from visible regions into masked areas. It produces temporally consistent fills for removed objects.

## How It Works

**References:**

- Paper: "FlowLens: Seeing Beyond the FoV via Optical Flow Completion" (Xu et al., ECCV 2022)

FlowLens decouples motion estimation from pixel synthesis by first completing optical flow
in masked regions, then using the completed flow for temporal propagation of known pixels,
followed by a refinement network for remaining holes, achieving sharp and temporally
consistent inpainting.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `FlowLens(NeuralNetworkArchitecture<>,FlowLensOptions,IGradientBasedOptimizer<,Tensor<>,Tensor<>>)` | Creates a FlowLens model for native training and inference. |
| `FlowLens(NeuralNetworkArchitecture<>,String,FlowLensOptions)` | Creates a FlowLens model for ONNX inference. |

## Methods

| Method | Summary |
|:-----|:--------|
| `CreateNewInstance` |  |
| `DeserializeNetworkSpecificData(BinaryReader)` |  |
| `Dispose(Boolean)` |  |
| `GetModelMetadata` |  |
| `GetOptions` |  |
| `InitializeLayers` |  |
| `Inpaint(Tensor<>,Tensor<>)` |  |
| `PostprocessOutput(Tensor<>)` |  |
| `PreprocessFrames(Tensor<>)` |  |
| `SerializeNetworkSpecificData(BinaryWriter)` |  |
| `Train(Tensor<>,Tensor<>)` |  |
| `UpdateParameters(Vector<>)` |  |

