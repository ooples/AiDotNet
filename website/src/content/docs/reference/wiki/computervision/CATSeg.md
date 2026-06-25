---
title: "CATSeg<T>"
description: "CAT-Seg: Cost Aggregation for open-vocabulary semantic segmentation."
section: "API Reference"
---

`Models & Types` · `AiDotNet.ComputerVision.Segmentation.OpenVocabulary`

CAT-Seg: Cost Aggregation for open-vocabulary semantic segmentation.

## For Beginners

Open-vocabulary segmentation via cost aggregation. Novel category recognition.

Common use cases:

- Open-vocabulary segmentation via cost aggregation
- Novel category recognition
- Cross-dataset segmentation
- Language-driven pixel classification

## How It Works

**Technical Details:**

- Cost volume between CLIP image and text features
- Cost aggregation transformer for spatial refinement
- Exploits CLIP feature similarity for pixel classification
- Efficient inference with frozen CLIP backbone

**Reference:** Cho et al., "CAT-Seg: Cost Aggregation for Open-Vocabulary Semantic Segmentation", CVPR 2024.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `CATSeg(NeuralNetworkArchitecture<>,IGradientBasedOptimizer<,Tensor<>,Tensor<>>,ILossFunction<>,Int32,Double,CATSegOptions)` | Initializes CATSeg in native (trainable) mode. |
| `CATSeg(NeuralNetworkArchitecture<>,String,Int32,CATSegOptions)` | Initializes CATSeg in ONNX (inference-only) mode. |

## Properties

| Property | Summary |
|:-----|:--------|
| `SupportsTraining` | Gets whether this CATSeg instance supports training. |

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

