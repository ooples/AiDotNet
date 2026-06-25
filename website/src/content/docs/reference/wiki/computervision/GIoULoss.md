---
title: "GIoULoss<T>"
description: "Generalized Intersection over Union (GIoU) loss for bounding box regression."
section: "API Reference"
---

`Models & Types` · `AiDotNet.ComputerVision.Detection.Losses`

Generalized Intersection over Union (GIoU) loss for bounding box regression.

## For Beginners

GIoU loss improves upon standard IoU loss by providing
gradients even when boxes don't overlap. This helps the model learn to move boxes
towards their targets even when they start far apart.

## How It Works

GIoU = IoU - (|C - U|) / |C|, where C is the smallest enclosing box and U is the union.
GIoU Loss = 1 - GIoU

Reference: Rezatofighi et al., "Generalized Intersection over Union: A Metric and A Loss
for Bounding Box Regression", CVPR 2019

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `GIoULoss` | Creates a new GIoU loss instance. |

## Methods

| Method | Summary |
|:-----|:--------|
| `CalculateDerivative(Vector<>,Vector<>)` | Calculates the gradient of GIoU loss with respect to predicted boxes. |
| `CalculateLoss(Tensor<>,Tensor<>)` | Calculates GIoU loss for tensor inputs. |
| `CalculateLoss(Vector<>,Vector<>)` | Calculates the GIoU loss between predicted and target bounding box vectors. |
| `CalculateLossForBox(BoundingBox<>,BoundingBox<>)` | Calculates GIoU loss for a pair of bounding boxes. |
| `ComputeTapeLoss(Tensor<>,Tensor<>)` |  |

