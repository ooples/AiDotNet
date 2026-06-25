---
title: "VMamba<T>"
description: "VMamba: Visual State Space Model with Cross-Scan for 2D understanding."
section: "API Reference"
---

`Models & Types` · `AiDotNet.ComputerVision.Segmentation.Mamba`

VMamba: Visual State Space Model with Cross-Scan for 2D understanding.

## For Beginners

Hierarchical visual feature extraction. Semantic segmentation backbone.

Common use cases:

- Hierarchical visual feature extraction
- Semantic segmentation backbone
- Dense prediction with global receptive field
- Efficient alternative to Swin Transformer

## How It Works

**Technical Details:**

- Cross-Scan Module (CSM) for 2D-aware state space scanning
- 4-way scanning to capture all spatial relationships
- Hierarchical architecture similar to Swin Transformer
- Linear complexity with global effective receptive field

**Reference:** Liu et al., "VMamba: Visual State Space Model", NeurIPS 2024.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `VMamba(NeuralNetworkArchitecture<>,IGradientBasedOptimizer<,Tensor<>,Tensor<>>,ILossFunction<>,Int32,VMambaModelSize,Double,VMambaOptions)` | Initializes VMamba in native (trainable) mode. |
| `VMamba(NeuralNetworkArchitecture<>,String,Int32,VMambaModelSize,VMambaOptions)` | Initializes VMamba in ONNX (inference-only) mode. |

## Properties

| Property | Summary |
|:-----|:--------|
| `SupportsTraining` | Gets whether this VMamba instance supports training. |

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

