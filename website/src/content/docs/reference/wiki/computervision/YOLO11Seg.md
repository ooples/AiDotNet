---
title: "YOLO11Seg<T>"
description: "YOLO11-Seg: Ultralytics next-generation real-time instance segmentation."
section: "API Reference"
---

`Models & Types` · `AiDotNet.ComputerVision.Segmentation.InstanceSegmentation`

YOLO11-Seg: Ultralytics next-generation real-time instance segmentation.

## For Beginners

Real-time instance segmentation. Edge deployment with INT8 quantization.

Common use cases:

- Real-time instance segmentation
- Edge deployment with INT8 quantization
- Video analytics pipelines
- Mobile and embedded applications

## How It Works

**Technical Details:**

- C2PSA (Cross-Stage Partial with Spatial Attention) blocks
- Improved feature pyramid for multi-scale detection
- Anchor-free decoupled head with mask branch
- YOLACT-style prototype mask generation

**Reference:** Ultralytics, "YOLO11", 2024.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `YOLO11Seg(NeuralNetworkArchitecture<>,IGradientBasedOptimizer<,Tensor<>,Tensor<>>,ILossFunction<>,Int32,YOLO11SegModelSize,Double,YOLO11SegOptions)` | Initializes YOLO11Seg in native (trainable) mode. |
| `YOLO11Seg(NeuralNetworkArchitecture<>,String,Int32,YOLO11SegModelSize,YOLO11SegOptions)` | Initializes YOLO11Seg in ONNX (inference-only) mode. |

## Properties

| Property | Summary |
|:-----|:--------|
| `SupportsTraining` | Gets whether this YOLO11Seg instance supports training. |

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

