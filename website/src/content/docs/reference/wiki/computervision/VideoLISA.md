---
title: "VideoLISA<T>"
description: "Video-LISA: Language-instructed video segmentation with reasoning."
section: "API Reference"
---

`Models & Types` · `AiDotNet.ComputerVision.Segmentation.Referring`

Video-LISA: Language-instructed video segmentation with reasoning.

## For Beginners

Language-instructed video segmentation. Temporal reasoning in video segmentation.

Common use cases:

- Language-instructed video segmentation
- Temporal reasoning in video segmentation
- Referring video object segmentation
- Conversational video understanding

## How It Works

**Technical Details:**

- Extends LISA to video with sparse token memory
- Sparse Dense Sampling for temporal video understanding
- One-Token-Seg-All: single token triggers multi-frame masks
- Handles temporal reasoning queries in video

**Reference:** Bai et al., "Video-LISA: Language Instructed Segmentation Assistant for Video", arXiv 2024.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `VideoLISA(NeuralNetworkArchitecture<>,IGradientBasedOptimizer<,Tensor<>,Tensor<>>,ILossFunction<>,Int32,Double,VideoLISAOptions)` | Initializes VideoLISA in native (trainable) mode. |
| `VideoLISA(NeuralNetworkArchitecture<>,String,Int32,VideoLISAOptions)` | Initializes VideoLISA in ONNX (inference-only) mode. |

## Properties

| Property | Summary |
|:-----|:--------|
| `SupportsTraining` | Gets whether this VideoLISA instance supports training. |

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

