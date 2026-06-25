---
title: "CIoULoss<T>"
description: "Complete Intersection over Union (CIoU) loss for bounding box regression."
section: "API Reference"
---

`Models & Types` · `AiDotNet.ComputerVision.Detection.Losses`

Complete Intersection over Union (CIoU) loss for bounding box regression.

## For Beginners

CIoU loss extends DIoU by also considering aspect ratio.
This provides the most accurate bounding box regression and is used in modern
YOLO versions (v5, v7, v8, etc.).

## How It Works

CIoU = IoU - d²/c² - αv, where:

- d is the center distance
- c is the enclosing diagonal
- v measures aspect ratio consistency
- α is a balancing factor

CIoU Loss = 1 - CIoU

Reference: Zheng et al., "Distance-IoU Loss: Faster and Better Learning for
Bounding Box Regression", AAAI 2020

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `CIoULoss` | Creates a new CIoU loss instance. |

## Methods

| Method | Summary |
|:-----|:--------|
| `CalculateDerivative(Vector<>,Vector<>)` | Calculates the gradient of CIoU loss with respect to predicted boxes. |
| `CalculateLoss(Tensor<>,Tensor<>)` | Calculates the CIoU loss between predicted and target bounding boxes. |
| `CalculateLoss(Vector<>,Vector<>)` | Calculates the CIoU loss between predicted and target bounding box vectors. |
| `CalculateLossForBox(BoundingBox<>,BoundingBox<>)` | Calculates CIoU loss for a pair of bounding boxes. |
| `ComputeTapeLoss(Tensor<>,Tensor<>)` |  |
| `ExtractBox(Tensor<>,Int32,Int32)` | Extracts a bounding box from a tensor at the specified indices. |

