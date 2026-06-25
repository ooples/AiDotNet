---
title: "MedSAM2<T>"
description: "MedSAM-2: SAM 2 adapted for medical image and video segmentation."
section: "API Reference"
---

`Models & Types` · `AiDotNet.ComputerVision.Segmentation.Medical`

MedSAM-2: SAM 2 adapted for medical image and video segmentation.

## For Beginners

Medical video segmentation (endoscopy, ultrasound). 3D volumetric medical segmentation (treat slices as video).

Common use cases:

- Medical video segmentation (endoscopy, ultrasound)
- 3D volumetric medical segmentation (treat slices as video)
- One-click medical segmentation across frames
- Temporal-consistent medical image analysis

## How It Works

**Technical Details:**

- SAM 2 architecture with memory attention for temporal consistency
- Treats 3D medical volumes as video sequences
- Point and box prompts propagated across frames/slices
- Hiera (Hierarchical) image encoder backbone

**Reference:** Zhu et al., "Medical SAM 2: Segment Medical Images As Video Via Segment Anything Model 2", arXiv 2024.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `MedSAM2(NeuralNetworkArchitecture<>,IGradientBasedOptimizer<,Tensor<>,Tensor<>>,ILossFunction<>,Int32,MedSAM2ModelSize,Double,MedSAM2Options)` | Initializes MedSAM2 in native (trainable) mode. |
| `MedSAM2(NeuralNetworkArchitecture<>,String,Int32,MedSAM2ModelSize,MedSAM2Options)` | Initializes MedSAM2 in ONNX (inference-only) mode. |

## Properties

| Property | Summary |
|:-----|:--------|
| `SupportsTraining` | Gets whether this MedSAM2 instance supports training. |

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

