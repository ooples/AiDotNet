---
title: "YOLO26Seg<T>"
description: "YOLO26-Seg: Latest YOLO family instance segmentation with advanced CSP design."
section: "API Reference"
---

`Models & Types` · `AiDotNet.ComputerVision.Segmentation.InstanceSegmentation`

YOLO26-Seg: Latest YOLO family instance segmentation with advanced CSP design.

## For Beginners

State-of-the-art real-time segmentation. Production deployment on GPU and edge.

Common use cases:

- State-of-the-art real-time segmentation
- Production deployment on GPU and edge
- Video stream processing
- Multi-object tracking with masks

## How It Works

**Technical Details:**

- Advanced Cross-Stage Partial design
- Improved feature pyramid network
- Anchor-free detection with mask prediction
- Optimized for TensorRT and ONNX export

**Reference:** Ultralytics, "YOLO26", 2025.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `YOLO26Seg(NeuralNetworkArchitecture<>,IGradientBasedOptimizer<,Tensor<>,Tensor<>>,ILossFunction<>,Int32,YOLO26SegModelSize,Double,YOLO26SegOptions)` | Initializes YOLO26Seg in native (trainable) mode. |
| `YOLO26Seg(NeuralNetworkArchitecture<>,String,Int32,YOLO26SegModelSize,YOLO26SegOptions)` | Initializes YOLO26Seg in ONNX (inference-only) mode. |

## Properties

| Property | Summary |
|:-----|:--------|
| `SupportsTraining` | Gets whether this YOLO26Seg instance supports training. |

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

