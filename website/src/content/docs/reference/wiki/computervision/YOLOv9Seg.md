---
title: "YOLOv9Seg<T>"
description: "YOLOv9-Seg: Instance segmentation with Programmable Gradient Information."
section: "API Reference"
---

`Models & Types` · `AiDotNet.ComputerVision.Segmentation.InstanceSegmentation`

YOLOv9-Seg: Instance segmentation with Programmable Gradient Information.

## For Beginners

YOLOv9-Seg extends the YOLOv9 object detector with a mask prediction
branch for real-time instance segmentation. It uses Programmable Gradient Information (PGI)
to preserve important information during training and GELAN (Generalized ELAN) for efficient
feature aggregation.

Common use cases:

- Real-time instance segmentation in video streams
- Autonomous driving (detecting and masking cars, pedestrians)
- Industrial inspection (defect detection with masks)
- Robotics (object manipulation with precise boundaries)

## How It Works

**Technical Details:**

- Programmable Gradient Information (PGI) prevents information loss in deep networks
- GELAN (Generalized ELAN) architecture for efficient feature aggregation
- Achieves 43.5 mask mAP on COCO with real-time speed
- Anchor-free detection head + YOLACT-style mask branch

**Reference:** Wang et al., "YOLOv9: Learning What You Want to Learn Using Programmable
Gradient Information", arXiv 2024.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `YOLOv9Seg(NeuralNetworkArchitecture<>,IGradientBasedOptimizer<,Tensor<>,Tensor<>>,ILossFunction<>,Int32,YOLOv9SegModelSize,Double,YOLOv9SegOptions)` | Initializes YOLOv9-Seg in native (trainable) mode. |
| `YOLOv9Seg(NeuralNetworkArchitecture<>,String,Int32,YOLOv9SegModelSize,YOLOv9SegOptions)` | Initializes YOLOv9-Seg in ONNX (inference-only) mode. |

## Properties

| Property | Summary |
|:-----|:--------|
| `SupportsTraining` | Gets whether this YOLOv9-Seg instance supports training. |

## Methods

| Method | Summary |
|:-----|:--------|
| `CreateNewInstance` | Creates a new instance with the same configuration but fresh weights. |
| `DeserializeNetworkSpecificData(BinaryReader)` | Reads configuration from a binary stream. |
| `Dispose(Boolean)` | Releases managed resources including the ONNX inference session. |
| `GetModelMetadata` | Collects metadata describing this model's configuration. |
| `InitializeLayers` | Initializes the backbone and detection/segmentation head layers. |
| `PredictCore(Tensor<>)` | Runs a forward pass for real-time instance segmentation. |
| `SerializeNetworkSpecificData(BinaryWriter)` | Writes configuration to a binary stream. |
| `Train(Tensor<>,Tensor<>)` | Performs one training step. |
| `UpdateParameters(Vector<>)` | Updates all trainable parameters from a flat parameter vector. |

