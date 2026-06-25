---
title: "NnUNet<T>"
description: "nnU-Net: Self-configuring framework for medical image segmentation."
section: "API Reference"
---

`Models & Types` · `AiDotNet.ComputerVision.Segmentation.Medical`

nnU-Net: Self-configuring framework for medical image segmentation.

## For Beginners

Medical image segmentation across organs and modalities. CT/MRI organ segmentation.

Common use cases:

- Medical image segmentation across organs and modalities
- CT/MRI organ segmentation
- Pathology image analysis
- Any biomedical segmentation task (self-configuring)

## How It Works

**Technical Details:**

- Self-configuring: automatically selects architecture, preprocessing, and training
- Supports 2D, 3D full-resolution, and 3D cascade configurations
- Rule-based pipeline configuration from dataset fingerprint
- Consistently top-ranked on medical segmentation benchmarks

**Reference:** Isensee et al., "nnU-Net: a self-configuring method for deep learning-based biomedical image segmentation", Nature Methods 2021.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `NnUNet(NeuralNetworkArchitecture<>,IGradientBasedOptimizer<,Tensor<>,Tensor<>>,ILossFunction<>,Int32,NnUNetModelSize,Double,NnUNetOptions)` | Initializes NnUNet in native (trainable) mode. |
| `NnUNet(NeuralNetworkArchitecture<>,String,Int32,NnUNetModelSize,NnUNetOptions)` | Initializes NnUNet in ONNX (inference-only) mode. |

## Properties

| Property | Summary |
|:-----|:--------|
| `SupportsTraining` | Gets whether this NnUNet instance supports training. |

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

