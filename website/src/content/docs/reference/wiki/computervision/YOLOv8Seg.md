---
title: "YOLOv8Seg<T>"
description: "YOLOv8-Seg: Ultralytics real-time instance segmentation."
section: "API Reference"
---

`Models & Types` · `AiDotNet.ComputerVision.Segmentation.InstanceSegmentation`

YOLOv8-Seg: Ultralytics real-time instance segmentation.

## For Beginners

YOLOv8-Seg is a fast, accurate instance segmentation model that detects
individual objects and produces per-pixel masks for each. It builds on the YOLO family's
anchor-free detection with a YOLACT-style prototype mask generation branch.

Common use cases:

- Real-time instance segmentation in video streams
- Object counting and measurement
- Autonomous driving perception
- Industrial quality inspection

## How It Works

**Technical Details:**

- CSPDarknet backbone with C2f (Cross-Stage Partial with 2 convolutions) blocks
- Anchor-free decoupled detection head
- YOLACT-style prototype masks with 32 coefficients
- Available in 5 sizes: N (3.4M), S (11.8M), M (27.3M), L (46.0M), X (71.8M)
- Input resolution: 640x640 by default

**Reference:** Ultralytics, "YOLOv8", 2023.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `YOLOv8Seg(NeuralNetworkArchitecture<>,IGradientBasedOptimizer<,Tensor<>,Tensor<>>,ILossFunction<>,Int32,YOLOv8SegModelSize,Double,YOLOv8SegOptions)` | Initializes YOLOv8Seg in native (trainable) mode. |
| `YOLOv8Seg(NeuralNetworkArchitecture<>,String,Int32,YOLOv8SegModelSize,YOLOv8SegOptions)` | Initializes YOLOv8Seg in ONNX (inference-only) mode. |

## Properties

| Property | Summary |
|:-----|:--------|
| `SupportsTraining` | Gets whether this YOLOv8Seg instance supports training. |

## Methods

| Method | Summary |
|:-----|:--------|
| `CreateNewInstance` | Creates a new instance with the same configuration. |
| `DeserializeNetworkSpecificData(BinaryReader)` | Reads configuration from a binary stream. |
| `Dispose(Boolean)` | Releases managed resources. |
| `GetModelMetadata` | Collects metadata describing this model's configuration. |
| `InitializeLayers` | Initializes the CSPDarknet encoder and detection + mask decoder layers. |
| `PredictCore(Tensor<>)` | Runs a forward pass to produce segmentation logits. |
| `SerializeNetworkSpecificData(BinaryWriter)` | Writes configuration to a binary stream. |
| `Train(Tensor<>,Tensor<>)` | Performs one training step. |
| `UpdateParameters(Vector<>)` | Updates all trainable parameters from a flat parameter vector. |

