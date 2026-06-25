---
title: "MaskRCNN<T>"
description: "Mask R-CNN for instance segmentation."
section: "API Reference"
---

`Models & Types` · `AiDotNet.ComputerVision.Segmentation.InstanceSegmentation`

Mask R-CNN for instance segmentation.

## For Beginners

Mask R-CNN extends Faster R-CNN by adding a mask
prediction branch parallel to the box classification and regression branches.
It's a two-stage detector that first proposes regions, then classifies them
and predicts masks.

## How It Works

Key features:

- Two-stage detection with RPN and RoI heads
- Parallel mask prediction branch
- RoIAlign for precise spatial alignment
- Decoupled mask and class prediction

Reference: He et al., "Mask R-CNN", ICCV 2017

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `MaskRCNN(InstanceSegmentationOptions<>)` | Creates a new Mask R-CNN model. |

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
| `SelectFPNLevel(BoundingBox<>,Int32)` | Selects the appropriate FPN level and stride based on proposal size. |

