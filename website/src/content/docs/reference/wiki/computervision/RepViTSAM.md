---
title: "RepViTSAM<T>"
description: "RepViT-SAM: Real-time SAM with RepViT backbone."
section: "API Reference"
---

`Models & Types` · `AiDotNet.ComputerVision.Segmentation.Efficient`

RepViT-SAM: Real-time SAM with RepViT backbone.

## For Beginners

Real-time segment anything. Mobile and edge SAM inference.

Common use cases:

- Real-time segment anything
- Mobile and edge SAM inference
- Interactive segmentation on devices
- Low-latency promptable segmentation

## How It Works

**Technical Details:**

- RepViT (Re-parameterized Vision Transformer) backbone
- Structural re-parameterization for inference speed
- Multi-scale feature extraction for SAM decoder
- Achieves real-time SAM on mobile devices

**Reference:** Wang et al., "RepViT-SAM: Towards Real-Time Segmenting Anything", arXiv 2024.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `RepViTSAM(NeuralNetworkArchitecture<>,IGradientBasedOptimizer<,Tensor<>,Tensor<>>,ILossFunction<>,Int32,Double,RepViTSAMOptions)` | Initializes RepViTSAM in native (trainable) mode. |
| `RepViTSAM(NeuralNetworkArchitecture<>,String,Int32,RepViTSAMOptions)` | Initializes RepViTSAM in ONNX (inference-only) mode. |

## Properties

| Property | Summary |
|:-----|:--------|
| `SupportsTraining` | Gets whether this RepViTSAM instance supports training. |

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

