---
title: "YOLOv11<T>"
description: "YOLOv11 object detector with enhanced feature extraction and attention mechanisms."
section: "API Reference"
---

`Models & Types` · `AiDotNet.ComputerVision.Detection.ObjectDetection.YOLO`

YOLOv11 object detector with enhanced feature extraction and attention mechanisms.

## For Beginners

YOLOv11 is the latest YOLO version with improved
feature extraction using attention mechanisms and more efficient architecture.
It builds upon YOLOv8-v10 innovations while adding new enhancements.

## How It Works

Key features:

- C3k2 blocks with attention for enhanced feature extraction
- Spatial Pyramid Pooling Fast (SPPF) with larger kernel
- Multi-head self-attention in neck
- Improved small object detection

Reference: Ultralytics, "YOLOv11" 2024

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `YOLOv11(ObjectDetectionOptions<>)` | Creates a new YOLOv11 detector. |

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

