---
title: "GroundedSAM2<T>"
description: "Grounded SAM 2: Text-grounded tracking and segmentation."
section: "API Reference"
---

`Models & Types` · `AiDotNet.ComputerVision.Segmentation.OpenVocabulary`

Grounded SAM 2: Text-grounded tracking and segmentation.

## For Beginners

Text-grounded video segmentation. Open-world object tracking.

Common use cases:

- Text-grounded video segmentation
- Open-world object tracking
- Natural language video search
- Automatic annotation from text descriptions

## How It Works

**Technical Details:**

- Grounding DINO for text-to-box detection + SAM 2 for segmentation
- Combines open-set detection with promptable segmentation
- Video tracking with text-specified targets
- Hiera backbone with memory attention for temporal consistency

**Reference:** Ren et al., "Grounded SAM: Assembling Open-World Models for Diverse Visual Tasks", arXiv 2024.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `GroundedSAM2(NeuralNetworkArchitecture<>,IGradientBasedOptimizer<,Tensor<>,Tensor<>>,ILossFunction<>,Int32,Double,GroundedSAM2Options)` | Initializes GroundedSAM2 in native (trainable) mode. |
| `GroundedSAM2(NeuralNetworkArchitecture<>,String,Int32,GroundedSAM2Options)` | Initializes GroundedSAM2 in ONNX (inference-only) mode. |

## Properties

| Property | Summary |
|:-----|:--------|
| `SupportsTraining` | Gets whether this GroundedSAM2 instance supports training. |

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

