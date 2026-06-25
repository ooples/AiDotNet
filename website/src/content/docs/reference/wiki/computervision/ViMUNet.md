---
title: "ViMUNet<T>"
description: "ViM-UNet: Vision Mamba for medical image segmentation in U-Net."
section: "API Reference"
---

`Models & Types` · `AiDotNet.ComputerVision.Segmentation.Mamba`

ViM-UNet: Vision Mamba for medical image segmentation in U-Net.

## For Beginners

Medical image segmentation with Mamba. Biomedical image analysis.

Common use cases:

- Medical image segmentation with Mamba
- Biomedical image analysis
- Efficient U-Net alternative
- Long-range dependency modeling in medical images

## How It Works

**Technical Details:**

- Vision Mamba encoder blocks in U-Net architecture
- Bidirectional SSM for capturing global context
- Skip connections between encoder and decoder
- Linear complexity for efficient medical image processing

**Reference:** Archit and Pape, "ViM-UNet: Vision Mamba for Biomedical Segmentation", arXiv 2024.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `ViMUNet(NeuralNetworkArchitecture<>,IGradientBasedOptimizer<,Tensor<>,Tensor<>>,ILossFunction<>,Int32,Double,ViMUNetOptions)` | Initializes ViMUNet in native (trainable) mode. |
| `ViMUNet(NeuralNetworkArchitecture<>,String,Int32,ViMUNetOptions)` | Initializes ViMUNet in ONNX (inference-only) mode. |

## Properties

| Property | Summary |
|:-----|:--------|
| `SupportsTraining` | Gets whether this ViMUNet instance supports training. |

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

