---
title: "UMamba<T>"
description: "U-Mamba: Hybrid CNN-Mamba architecture for medical segmentation."
section: "API Reference"
---

`Models & Types` · `AiDotNet.ComputerVision.Segmentation.Medical`

U-Mamba: Hybrid CNN-Mamba architecture for medical segmentation.

## For Beginners

Medical image segmentation with long-range dependencies. CT and MRI organ segmentation.

Common use cases:

- Medical image segmentation with long-range dependencies
- CT and MRI organ segmentation
- 3D medical volume segmentation
- Efficient medical AI with linear complexity

## How It Works

**Technical Details:**

- Hybrid CNN + Mamba (State Space Model) blocks in U-Net
- Linear complexity for processing long-range dependencies
- Captures both local CNN features and global SSM context
- U-Net architecture with Mamba blocks replacing transformer layers

**Reference:** Ma et al., "U-Mamba: Enhancing Long-range Dependency for Biomedical Image Segmentation", arXiv 2024.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `UMamba(NeuralNetworkArchitecture<>,IGradientBasedOptimizer<,Tensor<>,Tensor<>>,ILossFunction<>,Int32,Double,UMambaOptions)` | Initializes UMamba in native (trainable) mode. |
| `UMamba(NeuralNetworkArchitecture<>,String,Int32,UMambaOptions)` | Initializes UMamba in ONNX (inference-only) mode. |

## Properties

| Property | Summary |
|:-----|:--------|
| `SupportsTraining` | Gets whether this UMamba instance supports training. |

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

