---
title: "SwinUNETR<T>"
description: "Swin UNETR: Swin Transformer encoder for 3D medical segmentation."
section: "API Reference"
---

`Models & Types` · `AiDotNet.ComputerVision.Segmentation.Medical`

Swin UNETR: Swin Transformer encoder for 3D medical segmentation.

## For Beginners

3D medical volume segmentation. Brain tumor segmentation from MRI.

Common use cases:

- 3D medical volume segmentation
- Brain tumor segmentation from MRI
- CT organ segmentation
- Self-supervised pre-trained medical segmentation

## How It Works

**Technical Details:**

- Swin Transformer encoder with shifted window attention
- U-Net style decoder with skip connections from encoder stages
- Designed for 3D volumetric medical data
- Self-supervised pre-training on large medical datasets

**Reference:** Hatamizadeh et al., "Swin UNETR: Swin Transformers for Semantic Segmentation of Brain Tumors in MRI Images", BrainLes 2022.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `SwinUNETR(NeuralNetworkArchitecture<>,IGradientBasedOptimizer<,Tensor<>,Tensor<>>,ILossFunction<>,Int32,SwinUNETRModelSize,Double,SwinUNETROptions)` | Initializes SwinUNETR in native (trainable) mode. |
| `SwinUNETR(NeuralNetworkArchitecture<>,String,Int32,SwinUNETRModelSize,SwinUNETROptions)` | Initializes SwinUNETR in ONNX (inference-only) mode. |

## Properties

| Property | Summary |
|:-----|:--------|
| `SupportsTraining` | Gets whether this SwinUNETR instance supports training. |

## Methods

| Method | Summary |
|:-----|:--------|
| `CreateNewInstance` | Creates a new instance with the same configuration but fresh weights. |
| `DeserializeNetworkSpecificData(BinaryReader)` | Reads configuration from a binary stream. |
| `Dispose(Boolean)` | Releases managed resources including the ONNX inference session. |
| `GetModelMetadata` | Collects metadata describing this model's configuration. |
| `GetOrCreateBaseOptimizer` | Use the AdamW optimizer the constructor stored in `_optimizer` for the base class's tape-training path. |
| `InitializeLayers` | Initializes the encoder and decoder layers. |
| `PredictCore(Tensor<>)` | Runs a forward pass to produce segmentation logits. |
| `SerializeNetworkSpecificData(BinaryWriter)` | Writes configuration to a binary stream. |
| `Train(Tensor<>,Tensor<>)` | Performs one training step. |
| `UpdateParameters(Vector<>)` | Updates all trainable parameters from a flat parameter vector. |

