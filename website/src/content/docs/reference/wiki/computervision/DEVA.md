---
title: "DEVA<T>"
description: "DEVA: Tracking Anything with Decoupled Video Segmentation."
section: "API Reference"
---

`Models & Types` · `AiDotNet.ComputerVision.Segmentation.Video`

DEVA: Tracking Anything with Decoupled Video Segmentation.

## For Beginners

Video object segmentation and tracking. Open-world video segmentation.

Common use cases:

- Video object segmentation and tracking
- Open-world video segmentation
- Video editing and compositing
- Surveillance and activity monitoring

## How It Works

**Technical Details:**

- Decoupled image segmentation + temporal propagation
- Bi-directional temporal propagation module
- Works with any image segmentation model as front-end
- Online processing for real-time video segmentation

**Reference:** Cheng et al., "Tracking Anything with Decoupled Video Segmentation", ICCV 2023.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `DEVA(NeuralNetworkArchitecture<>,IGradientBasedOptimizer<,Tensor<>,Tensor<>>,ILossFunction<>,Int32,DEVAModelSize,Double,DEVAOptions)` | Initializes DEVA in native (trainable) mode. |
| `DEVA(NeuralNetworkArchitecture<>,String,Int32,DEVAModelSize,DEVAOptions)` | Initializes DEVA in ONNX (inference-only) mode. |

## Properties

| Property | Summary |
|:-----|:--------|
| `SupportsTraining` | Gets whether this DEVA instance supports training. |

## Methods

| Method | Summary |
|:-----|:--------|
| `CreateNewInstance` | Creates a new instance with the same configuration but fresh weights. |
| `DeserializeNetworkSpecificData(BinaryReader)` | Reads configuration from a binary stream. |
| `Dispose(Boolean)` | Releases managed resources including the ONNX inference session. |
| `GetModelMetadata` | Collects metadata describing this model's configuration. |
| `InitializeLayers` | Initializes the encoder and decoder layers. |
| `PredictCore(Tensor<>)` | Runs a forward pass to produce segmentation logits. |
| `SerializeNetworkSpecificData(BinaryWriter)` | Writes configuration to a binary stream. |
| `Train(Tensor<>,Tensor<>)` | Performs one training step. |
| `UpdateParameters(Vector<>)` | Updates all trainable parameters from a flat parameter vector. |

