---
title: "DETRSetLoss<T>"
description: "DETR Set Prediction Loss with Hungarian Matching for end-to-end object detection."
section: "API Reference"
---

`Models & Types` · `AiDotNet.ComputerVision.Detection.Losses`

DETR Set Prediction Loss with Hungarian Matching for end-to-end object detection.

## For Beginners

Unlike traditional detectors that use anchors and NMS,
DETR treats detection as a set prediction problem. It uses Hungarian matching to
find the optimal assignment between predicted and ground truth boxes, then computes
loss on the matched pairs.

## How It Works

The loss has three components:

- Classification loss: Cross-entropy for class predictions
- Box loss: L1 loss for box coordinates
- GIoU loss: For better box regression

Reference: Carion et al., "End-to-End Object Detection with Transformers", ECCV 2020

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `DETRSetLoss(Int32,Double,Double,Double)` | Creates a new DETR set loss instance. |

## Methods

| Method | Summary |
|:-----|:--------|
| `CalculateBoxGIoULoss(List<BoundingBox<>>,List<BoundingBox<>>,Int32[],Int32[])` | Calculates GIoU box loss for matched pairs. |
| `CalculateBoxL1Loss(List<BoundingBox<>>,List<BoundingBox<>>,Int32[],Int32[])` | Calculates L1 box loss for matched pairs. |
| `CalculateClassificationLoss(Double[0:,0:],List<Int32>,Int32[],Int32[],Int32)` | Calculates classification loss for matched pairs. |
| `CalculateDerivative(Tensor<>,Tensor<>)` | Calculates the gradient of the DETR loss. |
| `CalculateDerivative(Vector<>,Vector<>)` | Calculates the gradient of DETR loss with respect to predicted values. |
| `CalculateLoss(Tensor<>,Tensor<>)` | Calculates the DETR set loss. |
| `CalculateLoss(Vector<>,Vector<>)` | Calculates the DETR loss using flattened vectors (simplified for interface compatibility). |
| `ComputeClassCost(Double[0:,0:],Int32,Int32)` | Computes classification cost for Hungarian matching. |
| `ComputeL1Cost(BoundingBox<>,BoundingBox<>)` | Computes L1 cost between two boxes. |
| `ComputeTapeLoss(Tensor<>,Tensor<>)` |  |
| `ExtractGroundTruthBoxes(Tensor<>,Int32)` | Extracts ground truth boxes from the targets tensor. |
| `ExtractGroundTruthClasses(Tensor<>,Int32)` | Extracts ground truth class labels from the targets tensor. |
| `ExtractPredictedBoxes(Tensor<>,Int32,Int32)` | Extracts predicted boxes from the combined tensor. |
| `ExtractPredictedLogits(Tensor<>,Int32,Int32,Int32)` | Extracts predicted logits from the combined tensor. |
| `HungarianMatch(List<BoundingBox<>>,Double[0:,0:],List<BoundingBox<>>,List<Int32>)` | Performs Hungarian matching between predictions and ground truth. |

