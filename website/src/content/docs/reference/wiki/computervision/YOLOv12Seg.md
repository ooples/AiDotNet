---
title: "YOLOv12Seg<T>"
description: "YOLOv12-Seg: Attention-centric real-time instance segmentation."
section: "API Reference"
---

`Models & Types` · `AiDotNet.ComputerVision.Segmentation.InstanceSegmentation`

YOLOv12-Seg: Attention-centric real-time instance segmentation.

## For Beginners

Real-time instance segmentation with attention. Autonomous vehicle perception.

Common use cases:

- Real-time instance segmentation with attention
- Autonomous vehicle perception
- Smart surveillance systems
- Industrial quality control

## How It Works

**Technical Details:**

- Area-attention mechanism replacing standard self-attention
- R-ELAN (Residual Efficient Layer Aggregation Network) blocks
- FlashAttention-compatible efficient attention
- Native attention integration in YOLO architecture

**Reference:** Tian et al., "YOLOv12: Attention-Centric Real-Time Object Detectors", arXiv 2025.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `YOLOv12Seg(NeuralNetworkArchitecture<>,IGradientBasedOptimizer<,Tensor<>,Tensor<>>,ILossFunction<>,Int32,YOLOv12SegModelSize,Double,YOLOv12SegOptions)` | Initializes YOLOv12Seg in native (trainable) mode. |
| `YOLOv12Seg(NeuralNetworkArchitecture<>,String,Int32,YOLOv12SegModelSize,YOLOv12SegOptions)` | Initializes YOLOv12Seg in ONNX (inference-only) mode. |

## Properties

| Property | Summary |
|:-----|:--------|
| `SupportsTraining` | Gets whether this YOLOv12Seg instance supports training. |

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

