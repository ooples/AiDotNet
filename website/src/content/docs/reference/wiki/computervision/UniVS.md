---
title: "UniVS<T>"
description: "UniVS: Unified and Universal Video Segmentation."
section: "API Reference"
---

`Models & Types` · `AiDotNet.ComputerVision.Segmentation.Video`

UniVS: Unified and Universal Video Segmentation.

## For Beginners

Unified video segmentation across tasks. Video instance and panoptic segmentation.

Common use cases:

- Unified video segmentation across tasks
- Video instance and panoptic segmentation
- Referring video object segmentation
- Multi-task video understanding

## How It Works

**Technical Details:**

- Prompts-as-queries unifying VIS, VPS, and VOS
- Target-specific prompt generation
- Temporal KV memory for consistent tracking
- Handles both automatic and prompted segmentation

**Reference:** Li et al., "UniVS: Unified and Universal Video Segmentation with Prompts as Queries", CVPR 2024.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `UniVS(NeuralNetworkArchitecture<>,IGradientBasedOptimizer<,Tensor<>,Tensor<>>,ILossFunction<>,Int32,UniVSModelSize,Double,UniVSOptions)` | Initializes UniVS in native (trainable) mode. |
| `UniVS(NeuralNetworkArchitecture<>,String,Int32,UniVSModelSize,UniVSOptions)` | Initializes UniVS in ONNX (inference-only) mode. |

## Properties

| Property | Summary |
|:-----|:--------|
| `SupportsTraining` | Gets whether this UniVS instance supports training. |

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

