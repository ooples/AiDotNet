---
title: "CascadeRCNN<T>"
description: "Cascade R-CNN - Multi-stage object detection with progressive refinement."
section: "API Reference"
---

`Models & Types` · `AiDotNet.ComputerVision.Detection.ObjectDetection.RCNN`

Cascade R-CNN - Multi-stage object detection with progressive refinement.

## For Beginners

Cascade R-CNN improves upon Faster R-CNN by using multiple
detection stages, each trained with progressively higher IoU thresholds. This allows
the model to produce higher quality detections through progressive refinement.

## How It Works

Key features:

- Multiple cascade stages (typically 3) for progressive refinement
- Each stage uses higher IoU threshold for training
- Bounding boxes are refined at each stage
- Achieves higher accuracy than Faster R-CNN at slight speed cost

Reference: Cai and Vasconcelos, "Cascade R-CNN: Delving into High Quality Object Detection",
CVPR 2018

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `CascadeRCNN(ObjectDetectionOptions<>,Int32)` | Creates a new Cascade R-CNN detector. |

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

