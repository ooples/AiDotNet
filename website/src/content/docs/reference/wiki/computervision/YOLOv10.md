---
title: "YOLOv10<T>"
description: "YOLOv10 object detector with NMS-free training and consistent dual assignments."
section: "API Reference"
---

`Models & Types` · `AiDotNet.ComputerVision.Detection.ObjectDetection.YOLO`

YOLOv10 object detector with NMS-free training and consistent dual assignments.

## For Beginners

YOLOv10 eliminates the need for Non-Maximum Suppression (NMS)
during inference by using consistent dual assignments during training. This makes
inference faster and simpler while maintaining accuracy.

## How It Works

Key features:

- NMS-free inference via consistent dual assignments
- One-to-one matching during inference (no duplicate predictions)
- One-to-many matching for auxiliary training
- Reduced post-processing latency

Reference: Wang et al., "YOLOv10: Real-Time End-to-End Object Detection", 2024

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `YOLOv10(ObjectDetectionOptions<>,Boolean)` | Creates a new YOLOv10 detector. |

## Properties

| Property | Summary |
|:-----|:--------|
| `Name` |  |

## Methods

| Method | Summary |
|:-----|:--------|
| `Detect(Tensor<>,Double,Double)` |  |
| `Forward(Tensor<>)` |  |
| `GetHeadParameterCount` |  |
| `LoadWeightsAsync(String,CancellationToken)` |  |
| `PostProcess(List<Tensor<>>,Int32,Int32,Double,Double)` |  |
| `SaveWeights(String)` |  |
| `SelectTopKPerClass(List<Detection<>>,Int32)` | Selects top-K detections per class without NMS (for NMS-free inference). |

