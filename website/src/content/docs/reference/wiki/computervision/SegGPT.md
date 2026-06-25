---
title: "SegGPT<T>"
description: "SegGPT: Segmenting Everything In Context via in-context learning."
section: "API Reference"
---

`Models & Types` · `AiDotNet.ComputerVision.Segmentation.Interactive`

SegGPT: Segmenting Everything In Context via in-context learning.

## For Beginners

In-context segmentation from examples. Few-shot segmentation without fine-tuning.

Common use cases:

- In-context segmentation from examples
- Few-shot segmentation without fine-tuning
- Interactive image editing
- Versatile segmentation across domains

## How It Works

**Technical Details:**

- In-context learning for segmentation (no fine-tuning needed)
- ViT-Large backbone with random color mapping training
- Feature ensemble from multiple in-context examples
- Unified framework for semantic, instance, and part segmentation

**Reference:** Wang et al., "SegGPT: Segmenting Everything In Context", ICCV 2023.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `SegGPT(NeuralNetworkArchitecture<>,IGradientBasedOptimizer<,Tensor<>,Tensor<>>,ILossFunction<>,Int32,SegGPTModelSize,Double,SegGPTOptions)` | Initializes SegGPT in native (trainable) mode. |
| `SegGPT(NeuralNetworkArchitecture<>,String,Int32,SegGPTModelSize,SegGPTOptions)` | Initializes SegGPT in ONNX (inference-only) mode. |

## Properties

| Property | Summary |
|:-----|:--------|
| `SupportsTraining` | Gets whether this SegGPT instance supports training. |

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

