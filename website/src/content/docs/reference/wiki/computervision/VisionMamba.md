---
title: "VisionMamba<T>"
description: "Vision Mamba (Vim): Bidirectional State Space Model for vision."
section: "API Reference"
---

`Models & Types` · `AiDotNet.ComputerVision.Segmentation.Mamba`

Vision Mamba (Vim): Bidirectional State Space Model for vision.

## For Beginners

Efficient image segmentation with linear complexity. Dense prediction replacing ViT backbone.

Common use cases:

- Efficient image segmentation with linear complexity
- Dense prediction replacing ViT backbone
- Large-scale visual understanding
- Memory-efficient visual feature extraction

## How It Works

**Technical Details:**

- Bidirectional Mamba (SSM) for image sequence modeling
- Position-aware scanning (bi-directional for images)
- Linear complexity O(n) vs quadratic O(n^2) attention
- Competitive with ViT at significantly lower FLOPs

**Reference:** Zhu et al., "Vision Mamba: Efficient Visual Representation Learning with Bidirectional State Space Model", ICML 2024.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `VisionMamba(NeuralNetworkArchitecture<>,IGradientBasedOptimizer<,Tensor<>,Tensor<>>,ILossFunction<>,Int32,VisionMambaModelSize,Double,VisionMambaOptions)` | Initializes VisionMamba in native (trainable) mode. |
| `VisionMamba(NeuralNetworkArchitecture<>,String,Int32,VisionMambaModelSize,VisionMambaOptions)` | Initializes VisionMamba in ONNX (inference-only) mode. |

## Properties

| Property | Summary |
|:-----|:--------|
| `SupportsTraining` | Gets whether this VisionMamba instance supports training. |

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

