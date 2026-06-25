---
title: "PointTransformerV3<T>"
description: "Point Transformer V3: Simpler, Faster, Stronger 3D point cloud segmentation."
section: "API Reference"
---

`Models & Types` · `AiDotNet.ComputerVision.Segmentation.PointCloud`

Point Transformer V3: Simpler, Faster, Stronger 3D point cloud segmentation.

## For Beginners

3D point cloud semantic segmentation. Autonomous driving LiDAR processing.

Common use cases:

- 3D point cloud semantic segmentation
- Autonomous driving LiDAR processing
- Indoor scene understanding
- 3D object part segmentation

## How It Works

**Technical Details:**

- Serialized attention replacing expensive k-NN search
- Space-filling curves (Hilbert, Z-order) for point serialization
- Patch-based attention for scalability to large point clouds
- Supports both indoor and outdoor 3D segmentation

**Reference:** Wu et al., "Point Transformer V3: Simpler, Faster, Stronger", CVPR 2024.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `PointTransformerV3(NeuralNetworkArchitecture<>,IGradientBasedOptimizer<,Tensor<>,Tensor<>>,ILossFunction<>,Int32,PointTransformerV3ModelSize,Double,PointTransformerV3Options)` | Initializes PointTransformerV3 in native (trainable) mode. |
| `PointTransformerV3(NeuralNetworkArchitecture<>,String,Int32,PointTransformerV3ModelSize,PointTransformerV3Options)` | Initializes PointTransformerV3 in ONNX (inference-only) mode. |

## Properties

| Property | Summary |
|:-----|:--------|
| `SupportsTraining` | Gets whether this PointTransformerV3 instance supports training. |

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

