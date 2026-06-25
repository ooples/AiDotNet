---
title: "MaskAdapter<T>"
description: "Mask-Adapter: Adding SAM to open-vocabulary segmentation via mask prediction."
section: "API Reference"
---

`Models & Types` · `AiDotNet.ComputerVision.Segmentation.OpenVocabulary`

Mask-Adapter: Adding SAM to open-vocabulary segmentation via mask prediction.

## For Beginners

Enhanced open-vocabulary segmentation. SAM-guided mask proposals for open-vocab models.

Common use cases:

- Enhanced open-vocabulary segmentation
- SAM-guided mask proposals for open-vocab models
- Improved boundary quality in zero-shot segmentation
- Plug-and-play mask refinement

## How It Works

**Technical Details:**

- SAM as mask proposal generator for open-vocabulary segmentation
- Mask-level adaptation of CLIP features
- Plug-and-play adapter compatible with multiple backbones
- Improves both boundary quality and category accuracy

**Reference:** Xie et al., "Mask-Adapter: The Devil is in the Masks for Open-Vocabulary Segmentation", arXiv 2025.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `MaskAdapter(NeuralNetworkArchitecture<>,IGradientBasedOptimizer<,Tensor<>,Tensor<>>,ILossFunction<>,Int32,Double,MaskAdapterOptions)` | Initializes MaskAdapter in native (trainable) mode. |
| `MaskAdapter(NeuralNetworkArchitecture<>,String,Int32,MaskAdapterOptions)` | Initializes MaskAdapter in ONNX (inference-only) mode. |

## Properties

| Property | Summary |
|:-----|:--------|
| `SupportsTraining` | Gets whether this MaskAdapter instance supports training. |

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

