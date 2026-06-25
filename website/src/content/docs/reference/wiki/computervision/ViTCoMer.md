---
title: "ViTCoMer<T>"
description: "ViT-CoMer: Vision Transformer with Convolutional Multi-scale Feature Interaction."
section: "API Reference"
---

`Models & Types` · `AiDotNet.ComputerVision.Segmentation.Semantic`

ViT-CoMer: Vision Transformer with Convolutional Multi-scale Feature Interaction.

## For Beginners

ViT-CoMer is a hybrid model that combines a CNN branch with a Vision
Transformer branch, getting the best of both worlds. CNNs excel at capturing fine local details
(edges, textures), while transformers capture global context (relationships between distant objects).
By fusing them, ViT-CoMer produces segmentation maps with excellent boundary quality.

Common use cases:

- High-precision boundary segmentation (medical imaging, industrial inspection)
- Scene understanding where both local and global context matter
- Applications where ViTs alone miss fine details at object boundaries

## How It Works

**Technical Details:**

- Parallel CNN and transformer branches with cross-branch feature interaction
- CNN branch provides multi-scale local features at each ViT stage
- Bidirectional feature interaction module fuses CNN and transformer features
- Improved boundary quality over pure ViT or pure CNN approaches

**Reference:** Xia et al., "ViT-CoMer: Vision Transformer with Convolutional Multi-scale
Feature Interaction for Dense Predictions", CVPR 2024.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `ViTCoMer(NeuralNetworkArchitecture<>,IGradientBasedOptimizer<,Tensor<>,Tensor<>>,ILossFunction<>,Int32,ViTCoMerModelSize,Double,ViTCoMerOptions)` | Initializes a new instance of ViT-CoMer in native (trainable) mode. |
| `ViTCoMer(NeuralNetworkArchitecture<>,String,Int32,ViTCoMerModelSize,ViTCoMerOptions)` | Initializes a new instance of ViT-CoMer in ONNX (inference-only) mode. |

## Properties

| Property | Summary |
|:-----|:--------|
| `SupportsTraining` | Gets whether this ViT-CoMer instance supports training. |

## Methods

| Method | Summary |
|:-----|:--------|
| `CreateNewInstance` | Creates a new ViT-CoMer with same config but fresh weights. |
| `DeserializeNetworkSpecificData(BinaryReader)` | Deserializes configuration. |
| `Dispose(Boolean)` | Releases managed resources. |
| `GetModelMetadata` | Collects model metadata. |
| `GetOptions` | Gets the configuration options for this ViT-CoMer model. |
| `InitializeLayers` | Initializes the hybrid CNN-transformer encoder and decoder layers. |
| `PredictCore(Tensor<>)` | Runs a forward pass through the hybrid CNN-transformer model. |
| `SerializeNetworkSpecificData(BinaryWriter)` | Serializes configuration for persistence. |
| `Train(Tensor<>,Tensor<>)` | Performs one training step. |
| `UpdateParameters(Vector<>)` | Updates all trainable parameters from a flat vector. |

