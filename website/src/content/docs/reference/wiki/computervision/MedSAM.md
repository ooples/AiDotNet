---
title: "MedSAM<T>"
description: "MedSAM: Segment Anything in Medical Images."
section: "API Reference"
---

`Models & Types` · `AiDotNet.ComputerVision.Segmentation.Medical`

MedSAM: Segment Anything in Medical Images.

## For Beginners

Universal medical image segmentation. CT, MRI, ultrasound, X-ray, endoscopy segmentation.

Common use cases:

- Universal medical image segmentation
- CT, MRI, ultrasound, X-ray, endoscopy segmentation
- Box-prompted medical segmentation
- Clinical decision support

## How It Works

**Technical Details:**

- SAM architecture fine-tuned on 1.5M medical image-mask pairs
- ViT-Base image encoder + prompt encoder + mask decoder
- Supports bounding box prompts for medical ROI
- Covers 10+ medical imaging modalities

**Reference:** Ma et al., "Segment Anything in Medical Images", Nature Communications 2024.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `MedSAM(NeuralNetworkArchitecture<>,IGradientBasedOptimizer<,Tensor<>,Tensor<>>,ILossFunction<>,Int32,MedSAMModelSize,Double,MedSAMOptions)` | Initializes MedSAM in native (trainable) mode. |
| `MedSAM(NeuralNetworkArchitecture<>,String,Int32,MedSAMModelSize,MedSAMOptions)` | Initializes MedSAM in ONNX (inference-only) mode. |

## Properties

| Property | Summary |
|:-----|:--------|
| `SupportsTraining` | Gets whether this MedSAM instance supports training. |

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

