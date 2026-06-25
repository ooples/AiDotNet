---
title: "GaVS<T>"
description: "GaVS gaze-aware video stabilization with saliency-weighted motion smoothing."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Video.Stabilization`

GaVS gaze-aware video stabilization with saliency-weighted motion smoothing.

## For Beginners

GaVS (Generative Adversarial Video Stabilization) uses adversarial training to produce stabilized video that looks natural. The discriminator ensures the output appears like genuinely stable footage.

## How It Works

**References:**

- Paper: "Gaze-aware Video Stabilization" (2023)

GaVS predicts viewer gaze regions and applies stronger stabilization near the focus of
attention while allowing more camera motion in peripheral regions. This preserves
intentional cinematographic movements while removing distracting shake near gaze targets.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `GaVS(NeuralNetworkArchitecture<>,GaVSOptions,IGradientBasedOptimizer<,Tensor<>,Tensor<>>)` | Creates a GaVS model for native training and inference. |
| `GaVS(NeuralNetworkArchitecture<>,String,GaVSOptions)` | Creates a GaVS model for ONNX inference. |

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

