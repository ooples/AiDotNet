---
title: "CUPS<T>"
description: "CUPS: Unified Panoptic Segmentation with Comprehensive Use of Pixels and Semantics."
section: "API Reference"
---

`Models & Types` · `AiDotNet.ComputerVision.Segmentation.Panoptic`

CUPS: Unified Panoptic Segmentation with Comprehensive Use of Pixels and Semantics.

## For Beginners

Unified panoptic segmentation. Semantic-aware scene understanding.

Common use cases:

- Unified panoptic segmentation
- Semantic-aware scene understanding
- Comprehensive pixel-level labeling
- Autonomous driving perception

## How It Works

**Technical Details:**

- Unified pixel-semantic architecture
- Joint stuff and thing segmentation
- Efficient feature fusion from multiple scales
- Swin Transformer backbone

**Reference:** de Geus et al., "CUPS: Comprehensive Use of Pixels and Semantics for Panoptic Segmentation", arXiv 2023.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `CUPS(NeuralNetworkArchitecture<>,IGradientBasedOptimizer<,Tensor<>,Tensor<>>,ILossFunction<>,Int32,Double,CUPSOptions)` | Initializes CUPS in native (trainable) mode. |
| `CUPS(NeuralNetworkArchitecture<>,String,Int32,CUPSOptions)` | Initializes CUPS in ONNX (inference-only) mode. |

## Properties

| Property | Summary |
|:-----|:--------|
| `SupportsTraining` | Gets whether this CUPS instance supports training. |

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

