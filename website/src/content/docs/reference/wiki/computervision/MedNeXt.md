---
title: "MedNeXt<T>"
description: "MedNeXt: Transformer-driven scaling of ConvNets for medical segmentation."
section: "API Reference"
---

`Models & Types` · `AiDotNet.ComputerVision.Segmentation.Medical`

MedNeXt: Transformer-driven scaling of ConvNets for medical segmentation.

## For Beginners

Medical image segmentation with efficient ConvNet design. CT and MRI organ segmentation.

Common use cases:

- Medical image segmentation with efficient ConvNet design
- CT and MRI organ segmentation
- 3D medical volume analysis
- Resource-efficient medical AI deployment

## How It Works

**Technical Details:**

- ConvNeXt-inspired blocks adapted for medical imaging
- Large kernel sizes (up to 7x7x7) for capturing global context
- UpKern: compound scaling strategy for depth, width, and kernel size
- Achieves transformer-level performance with pure convolutions

**Reference:** Roy et al., "MedNeXt: Transformer-driven Scaling of ConvNets for Medical Image Segmentation", MICCAI 2023.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `MedNeXt(NeuralNetworkArchitecture<>,IGradientBasedOptimizer<,Tensor<>,Tensor<>>,ILossFunction<>,Int32,MedNeXtModelSize,Double,MedNeXtOptions)` | Initializes MedNeXt in native (trainable) mode. |
| `MedNeXt(NeuralNetworkArchitecture<>,String,Int32,MedNeXtModelSize,MedNeXtOptions)` | Initializes MedNeXt in ONNX (inference-only) mode. |

## Properties

| Property | Summary |
|:-----|:--------|
| `SupportsTraining` | Gets whether this MedNeXt instance supports training. |

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

