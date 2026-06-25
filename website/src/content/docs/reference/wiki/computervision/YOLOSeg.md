---
title: "YOLOSeg<T>"
description: "YOLOv8-Seg for instance segmentation."
section: "API Reference"
---

`Models & Types` · `AiDotNet.ComputerVision.Segmentation.InstanceSegmentation`

YOLOv8-Seg for instance segmentation.

## For Beginners

YOLOv8-Seg extends YOLOv8 detection with a segmentation head.
It uses prototype masks and per-instance coefficients for efficient mask prediction.
This is much faster than Mask R-CNN while maintaining good quality.

## How It Works

Key features:

- Single-stage detection + segmentation
- Prototype-based mask assembly
- Shared backbone with detection head
- Real-time performance

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `YOLOSeg(InstanceSegmentationOptions<>)` | Creates a new YOLOv8-Seg model. |

## Properties

| Property | Summary |
|:-----|:--------|
| `Name` |  |

## Methods

| Method | Summary |
|:-----|:--------|
| `GetParameterCount` |  |
| `LoadWeightsAsync(String,CancellationToken)` |  |
| `SaveWeights(String)` |  |
| `Segment(Tensor<>)` |  |

