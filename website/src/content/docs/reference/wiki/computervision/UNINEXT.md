---
title: "UNINEXT<T>"
description: "UNINEXT: Universal Instance Perception as Object Discovery and Retrieval."
section: "API Reference"
---

`Models & Types` · `AiDotNet.ComputerVision.Segmentation.Foundation`

UNINEXT: Universal Instance Perception as Object Discovery and Retrieval.

## For Beginners

UNINEXT reformulates over 10 different instance perception tasks into a
unified "discover and retrieve" framework. Whether you need object detection, instance segmentation,
single-object tracking, multi-object tracking, video instance segmentation, or referring expression
segmentation, UNINEXT handles them all with one model by using task-specific prompt embeddings.

Common use cases:

- Multi-task instance perception (detection + segmentation + tracking)
- Video object segmentation with tracking
- Referring expression comprehension (find object from text description)
- SOTA results on 20+ benchmarks simultaneously

## How It Works

**Technical Details:**

- Unified query representation for all instance perception tasks
- Task-specific prompt embeddings select which task to perform
- Backbone: ResNet-50, Swin-L, or ViT-H
- Deformable transformer encoder-decoder
- SOTA on 20+ benchmarks across detection, segmentation, tracking

**Reference:** Yan et al., "Universal Instance Perception as Object Discovery and Retrieval",
CVPR 2023.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `UNINEXT(NeuralNetworkArchitecture<>,IGradientBasedOptimizer<,Tensor<>,Tensor<>>,ILossFunction<>,Int32,Int32,UNINEXTModelSize,Double,UNINEXTOptions)` | Initializes UNINEXT in native (trainable) mode. |
| `UNINEXT(NeuralNetworkArchitecture<>,String,Int32,Int32,UNINEXTModelSize,UNINEXTOptions)` | Initializes UNINEXT in ONNX (inference-only) mode. |

## Properties

| Property | Summary |
|:-----|:--------|
| `SupportsTraining` | Gets whether this UNINEXT instance supports training. |

## Methods

| Method | Summary |
|:-----|:--------|
| `CreateNewInstance` | Creates a new UNINEXT instance with the same configuration but fresh weights. |
| `DeserializeNetworkSpecificData(BinaryReader)` | Reads UNINEXT configuration from a binary stream. |
| `Dispose(Boolean)` | Releases managed resources including the ONNX inference session. |
| `GetModelMetadata` | Collects metadata describing this UNINEXT model's configuration. |
| `GetOptions` | Gets the configuration options for this UNINEXT model. |
| `InitializeLayers` | Initializes the encoder and decoder layers for UNINEXT. |
| `PredictCore(Tensor<>)` | Runs a forward pass through UNINEXT for unified instance perception. |
| `SerializeNetworkSpecificData(BinaryWriter)` | Writes UNINEXT configuration to a binary stream. |
| `Train(Tensor<>,Tensor<>)` | Performs one training step. |
| `UpdateParameters(Vector<>)` | Updates all trainable parameters from a flat parameter vector. |

