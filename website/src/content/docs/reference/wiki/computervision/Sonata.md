---
title: "Sonata<T>"
description: "Sonata: A Mamba-based 3D point cloud backbone for efficient segmentation."
section: "API Reference"
---

`Models & Types` · `AiDotNet.ComputerVision.Segmentation.PointCloud`

Sonata: A Mamba-based 3D point cloud backbone for efficient segmentation.

## For Beginners

Efficient 3D point cloud segmentation. Large-scale LiDAR scene understanding.

Common use cases:

- Efficient 3D point cloud segmentation
- Large-scale LiDAR scene understanding
- Real-time 3D perception for robotics
- Memory-efficient 3D processing

## How It Works

**Technical Details:**

- Mamba (State Space Model) for linear-complexity point cloud processing
- Serialized point cloud input via space-filling curves
- Scales to millions of points without quadratic attention cost
- Competitive with transformer-based methods at lower compute

**Reference:** Wu et al., "Sonata and Concerto: Mamba for 3D Point Clouds", arXiv 2024.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `Sonata(NeuralNetworkArchitecture<>,IGradientBasedOptimizer<,Tensor<>,Tensor<>>,ILossFunction<>,Int32,SonataModelSize,Double,SonataOptions)` | Initializes Sonata in native (trainable) mode. |
| `Sonata(NeuralNetworkArchitecture<>,String,Int32,SonataModelSize,SonataOptions)` | Initializes Sonata in ONNX (inference-only) mode. |

## Properties

| Property | Summary |
|:-----|:--------|
| `SupportsTraining` | Gets whether this Sonata instance supports training. |

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

