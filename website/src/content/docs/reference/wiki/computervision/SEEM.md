---
title: "SEEM<T>"
description: "SEEM: Segment Everything Everywhere All at Once."
section: "API Reference"
---

`Models & Types` · `AiDotNet.ComputerVision.Segmentation.Interactive`

SEEM: Segment Everything Everywhere All at Once.

## For Beginners

Multi-modal interactive segmentation. Text-guided, click-guided, and box-guided segmentation.

Common use cases:

- Multi-modal interactive segmentation
- Text-guided, click-guided, and box-guided segmentation
- Referring expression segmentation
- Open-vocabulary segmentation

## How It Works

**Technical Details:**

- Joint visual-semantic decoder for multi-modal prompts
- Supports text, point, box, and scribble prompts simultaneously
- Focal-Tiny or Focal-Large backbone
- Unified architecture for interactive + automatic segmentation

**Reference:** Zou et al., "Segment Everything Everywhere All at Once", NeurIPS 2023.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `SEEM(NeuralNetworkArchitecture<>,IGradientBasedOptimizer<,Tensor<>,Tensor<>>,ILossFunction<>,Int32,SEEMModelSize,Double,SEEMOptions)` | Initializes SEEM in native (trainable) mode. |
| `SEEM(NeuralNetworkArchitecture<>,String,Int32,SEEMModelSize,SEEMOptions)` | Initializes SEEM in ONNX (inference-only) mode. |

## Properties

| Property | Summary |
|:-----|:--------|
| `SupportsTraining` | Gets whether this SEEM instance supports training. |

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

