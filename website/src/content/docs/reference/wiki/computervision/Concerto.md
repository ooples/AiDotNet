---
title: "Concerto<T>"
description: "Concerto: Hybrid Mamba-Transformer backbone for 3D point clouds."
section: "API Reference"
---

`Models & Types` · `AiDotNet.ComputerVision.Segmentation.PointCloud`

Concerto: Hybrid Mamba-Transformer backbone for 3D point clouds.

## For Beginners

High-accuracy 3D point cloud segmentation. Multi-modal 3D scene understanding.

Common use cases:

- High-accuracy 3D point cloud segmentation
- Multi-modal 3D scene understanding
- Dense 3D prediction tasks
- Advanced LiDAR perception

## How It Works

**Technical Details:**

- Hybrid Mamba + Transformer blocks for balanced efficiency and accuracy
- Global Mamba branches + local Transformer attention
- Better quality than pure Mamba while remaining efficient
- Serialized point processing with space-filling curves

**Reference:** Wu et al., "Sonata and Concerto: Mamba for 3D Point Clouds", arXiv 2024.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `Concerto(NeuralNetworkArchitecture<>,IGradientBasedOptimizer<,Tensor<>,Tensor<>>,ILossFunction<>,Int32,ConcertoModelSize,Double,ConcertoOptions)` | Initializes Concerto in native (trainable) mode. |
| `Concerto(NeuralNetworkArchitecture<>,String,Int32,ConcertoModelSize,ConcertoOptions)` | Initializes Concerto in ONNX (inference-only) mode. |

## Properties

| Property | Summary |
|:-----|:--------|
| `SupportsTraining` | Gets whether this Concerto instance supports training. |

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

