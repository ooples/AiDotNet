---
title: "NMS<T>"
description: "Implements Non-Maximum Suppression (NMS) algorithms for removing duplicate detections."
section: "API Reference"
---

`Models & Types` · `AiDotNet.ComputerVision.Detection.PostProcessing`

Implements Non-Maximum Suppression (NMS) algorithms for removing duplicate detections.

## For Beginners

When an object detector runs, it often produces multiple
overlapping bounding boxes for the same object. NMS removes these duplicates by:

1. Keeping the detection with highest confidence
2. Removing other detections that overlap too much with it
3. Repeating until all duplicates are removed

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `NMS` | Creates a new NMS instance. |

## Methods

| Method | Summary |
|:-----|:--------|
| `Apply(List<Detection<>>,Double)` | Performs standard NMS on a list of detections. |
| `ApplyBatched(List<List<Detection<>>>,Double,Boolean)` | Performs batched NMS for efficient processing of multiple images. |
| `ApplyClassAware(List<Detection<>>,Double)` | Performs class-aware NMS (applies NMS separately per class). |
| `ComputeCIoU(BoundingBox<>,BoundingBox<>)` | Computes Complete IoU (CIoU) between two bounding boxes. |
| `ComputeDIoU(BoundingBox<>,BoundingBox<>)` | Computes Distance IoU (DIoU) between two bounding boxes. |
| `ComputeGIoU(BoundingBox<>,BoundingBox<>)` | Computes Generalized IoU (GIoU) between two bounding boxes. |
| `ComputeIoU(BoundingBox<>,BoundingBox<>)` | Computes Intersection over Union (IoU) between two bounding boxes. |

