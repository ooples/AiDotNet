---
title: "STTN<T>"
description: "STTN spatial-temporal transformer network for video inpainting with multi-scale attention."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Video.Inpainting`

STTN spatial-temporal transformer network for video inpainting with multi-scale attention.

## For Beginners

STTN (Spatial-Temporal Transformer Network) performs video inpainting using transformers that jointly attend to spatial and temporal dimensions to fill holes consistently across frames.

## How It Works

**References:**

- Paper: "Learning Joint Spatial-Temporal Transformations for Video Inpainting" (Zeng et al., ECCV 2020)

STTN uses multi-scale spatial-temporal transformers that jointly search for and attend to
relevant patches across both space and time dimensions. Multi-head attention at multiple
feature scales enables both fine-grained texture transfer and large-scale structure completion.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `STTN(NeuralNetworkArchitecture<>,STTNOptions,IGradientBasedOptimizer<,Tensor<>,Tensor<>>)` | Creates a STTN model for native training and inference. |
| `STTN(NeuralNetworkArchitecture<>,String,STTNOptions)` | Creates a STTN model for ONNX inference. |

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

