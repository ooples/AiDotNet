---
title: "YOLOv8<T>"
description: "YOLOv8 object detector - anchor-free, decoupled head architecture."
section: "API Reference"
---

`Models & Types` · `AiDotNet.ComputerVision.Detection.ObjectDetection.YOLO`

YOLOv8 object detector - anchor-free, decoupled head architecture.

## For Beginners

YOLOv8 is a state-of-the-art real-time object detector
developed by Ultralytics. It's faster and more accurate than previous YOLO versions,
using an anchor-free design that simplifies training and improves generalization.

## How It Works

Key improvements over YOLOv5:

- Anchor-free design eliminates anchor tuning
- Decoupled head separates classification and localization
- Distribution Focal Loss (DFL) for better box regression
- C2f modules for improved feature extraction

Reference: Jocher et al., "YOLOv8" Ultralytics, 2023

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `YOLOv8(ObjectDetectionOptions<>)` | Creates a new YOLOv8 detector. |

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
| `GetSizeConfig(ModelSize)` | Gets depth and width multipliers for each model size. |
| `LoadWeightsAsync(String,CancellationToken)` |  |
| `PostProcess(List<Tensor<>>,Int32,Int32,Double,Double)` |  |
| `SaveWeights(String)` |  |

