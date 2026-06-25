---
title: "DIoULoss<T>"
description: "Distance Intersection over Union (DIoU) loss for bounding box regression."
section: "API Reference"
---

`Models & Types` · `AiDotNet.ComputerVision.Detection.Losses`

Distance Intersection over Union (DIoU) loss for bounding box regression.

## For Beginners

DIoU loss adds a center distance penalty to GIoU loss.
This helps the model converge faster by explicitly minimizing the distance between
predicted and target box centers.

## How It Works

DIoU = IoU - d²/c², where d is the center distance and c is the enclosing diagonal.
DIoU Loss = 1 - DIoU

Reference: Zheng et al., "Distance-IoU Loss: Faster and Better Learning for
Bounding Box Regression", AAAI 2020

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `DIoULoss` | Creates a new DIoU loss instance. |

## Methods

| Method | Summary |
|:-----|:--------|
| `CalculateDerivative(Vector<>,Vector<>)` | Calculates the gradient of DIoU loss with respect to predicted boxes. |
| `CalculateLoss(Tensor<>,Tensor<>)` | Calculates the DIoU loss between predicted and target bounding boxes. |
| `CalculateLoss(Vector<>,Vector<>)` | Calculates the DIoU loss between predicted and target bounding box vectors. |
| `CalculateLossForBox(BoundingBox<>,BoundingBox<>)` | Calculates DIoU loss for a pair of bounding boxes. |
| `ComputeTapeLoss(Tensor<>,Tensor<>)` |  |
| `ExtractBox(Tensor<>,Int32,Int32)` | Extracts a bounding box from a tensor at the specified indices. |

