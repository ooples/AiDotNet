---
title: "PIDNet<T>"
description: "PIDNet: A Real-time Semantic Segmentation Network Inspired by PID Controllers."
section: "API Reference"
---

`Models & Types` · `AiDotNet.ComputerVision.Segmentation.Efficient`

PIDNet: A Real-time Semantic Segmentation Network Inspired by PID Controllers.

## For Beginners

Real-time semantic segmentation for autonomous driving. Cityscapes and urban scene parsing.

Common use cases:

- Real-time semantic segmentation for autonomous driving
- Cityscapes and urban scene parsing
- Edge-device semantic segmentation
- Balanced speed-accuracy segmentation

## How It Works

**Technical Details:**

- PID-controller-inspired three-branch architecture (P, I, D)
- P-branch: detail features, I-branch: context, D-branch: boundary
- Bag of Tricks for efficient feature fusion
- Achieves Cityscapes mIoU 80+ at 90+ FPS

**Reference:** Xu et al., "PIDNet: A Real-time Semantic Segmentation Network Inspired by PID Controllers", CVPR 2023.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `PIDNet(NeuralNetworkArchitecture<>,IGradientBasedOptimizer<,Tensor<>,Tensor<>>,ILossFunction<>,Int32,PIDNetModelSize,Double,PIDNetOptions)` | Initializes PIDNet in native (trainable) mode. |
| `PIDNet(NeuralNetworkArchitecture<>,String,Int32,PIDNetModelSize,PIDNetOptions)` | Initializes PIDNet in ONNX (inference-only) mode. |

## Properties

| Property | Summary |
|:-----|:--------|
| `SupportsTraining` | Gets whether this PIDNet instance supports training. |

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

