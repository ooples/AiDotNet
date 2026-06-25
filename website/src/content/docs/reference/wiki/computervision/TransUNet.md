---
title: "TransUNet<T>"
description: "TransUNet: Transformers make strong encoders for medical segmentation."
section: "API Reference"
---

`Models & Types` · `AiDotNet.ComputerVision.Segmentation.Medical`

TransUNet: Transformers make strong encoders for medical segmentation.

## For Beginners

Multi-organ segmentation from CT scans. Cardiac segmentation from MRI.

Common use cases:

- Multi-organ segmentation from CT scans
- Cardiac segmentation from MRI
- Medical image analysis requiring global context
- Hybrid CNN-Transformer medical segmentation

## How It Works

**Technical Details:**

- Hybrid CNN-Transformer encoder (ResNet + ViT)
- CNN captures local features, Transformer captures global context
- Cascaded upsampler decoder with skip connections
- Tokenized image patches enable self-attention over full image

**Reference:** Chen et al., "TransUNet: Transformers Make Strong Encoders for Medical Image Segmentation", arXiv 2021.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `TransUNet(NeuralNetworkArchitecture<>,IGradientBasedOptimizer<,Tensor<>,Tensor<>>,ILossFunction<>,Int32,TransUNetModelSize,Double,TransUNetOptions)` | Initializes TransUNet in native (trainable) mode. |
| `TransUNet(NeuralNetworkArchitecture<>,String,Int32,TransUNetModelSize,TransUNetOptions)` | Initializes TransUNet in ONNX (inference-only) mode. |

## Properties

| Property | Summary |
|:-----|:--------|
| `SupportsTraining` | Gets whether this TransUNet instance supports training. |

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

