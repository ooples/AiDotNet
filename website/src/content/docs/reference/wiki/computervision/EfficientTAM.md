---
title: "EfficientTAM<T>"
description: "EfficientTAM: Efficient Track Anything Model for edge video segmentation."
section: "API Reference"
---

`Models & Types` · `AiDotNet.ComputerVision.Segmentation.Video`

EfficientTAM: Efficient Track Anything Model for edge video segmentation.

## For Beginners

Edge-device video segmentation. Mobile video object tracking.

Common use cases:

- Edge-device video segmentation
- Mobile video object tracking
- Real-time interactive video editing
- Low-latency video analytics

## How It Works

**Technical Details:**

- Lightweight encoder replacing SAM2 heavy backbone
- Efficient memory mechanism for temporal propagation
- Designed for mobile and edge deployment
- Compatible with SAM2 prompt interface

**Reference:** Yang et al., "EfficientTAM: Efficient Track Anything Model", arXiv 2024.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `EfficientTAM(NeuralNetworkArchitecture<>,IGradientBasedOptimizer<,Tensor<>,Tensor<>>,ILossFunction<>,Int32,EfficientTAMModelSize,Double,EfficientTAMOptions)` | Initializes EfficientTAM in native (trainable) mode. |
| `EfficientTAM(NeuralNetworkArchitecture<>,String,Int32,EfficientTAMModelSize,EfficientTAMOptions)` | Initializes EfficientTAM in ONNX (inference-only) mode. |

## Properties

| Property | Summary |
|:-----|:--------|
| `SupportsTraining` | Gets whether this EfficientTAM instance supports training. |

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

