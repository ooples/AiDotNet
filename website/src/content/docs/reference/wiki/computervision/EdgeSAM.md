---
title: "EdgeSAM<T>"
description: "EdgeSAM: Prompt-in-the-Loop Distillation for SAM on edge devices."
section: "API Reference"
---

`Models & Types` · `AiDotNet.ComputerVision.Segmentation.Efficient`

EdgeSAM: Prompt-in-the-Loop Distillation for SAM on edge devices.

## For Beginners

Edge-device segment anything. On-device interactive segmentation.

Common use cases:

- Edge-device segment anything
- On-device interactive segmentation
- Mobile and IoT segmentation
- Real-time promptable segmentation

## How It Works

**Technical Details:**

- RepViT encoder for edge-optimized inference
- Prompt-in-the-loop distillation from SAM ViT-H
- End-to-end encoder + decoder distillation
- Runs on iPhone 14 at interactive speeds

**Reference:** Zhou et al., "EdgeSAM: Prompt-In-the-Loop Distillation for On-Device Deployment of SAM", arXiv 2024.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `EdgeSAM(NeuralNetworkArchitecture<>,IGradientBasedOptimizer<,Tensor<>,Tensor<>>,ILossFunction<>,Int32,Double,EdgeSAMOptions)` | Initializes EdgeSAM in native (trainable) mode. |
| `EdgeSAM(NeuralNetworkArchitecture<>,String,Int32,EdgeSAMOptions)` | Initializes EdgeSAM in ONNX (inference-only) mode. |

## Properties

| Property | Summary |
|:-----|:--------|
| `SupportsTraining` | Gets whether this EdgeSAM instance supports training. |

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

