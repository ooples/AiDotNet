---
title: "IInstanceSegmentation<T>"
description: "Interface for instance segmentation models that detect and mask individual object instances."
section: "API Reference"
---

`Interfaces` · `AiDotNet.Interfaces`

Interface for instance segmentation models that detect and mask individual object instances.

## For Beginners

Instance segmentation answers "where is each individual object?"

Unlike semantic segmentation (which just says "these pixels are car"), instance segmentation
says "these pixels are car #1, those pixels are car #2, and those are car #3".

Each detection includes:

- A bounding box (rectangle around the object)
- A binary mask (exact pixel outline)
- A class label (what the object is)
- A confidence score (how sure the model is)

Models implementing this interface:

- YOLOv9-Seg, YOLO11-Seg, YOLOv12-Seg, YOLO26-Seg (real-time)
- Mask2Former, MaskDINO (transformer-based)

## How It Works

Instance segmentation combines object detection with pixel-level masking. Each detected
object gets its own binary mask, allowing you to distinguish between individual instances
of the same class (e.g., car #1 vs. car #2).

## Properties

| Property | Summary |
|:-----|:--------|
| `ConfidenceThreshold` | Gets or sets the confidence threshold for filtering detections. |
| `MaxInstances` | Gets the maximum number of instances the model can detect per image. |
| `NmsThreshold` | Gets or sets the IoU threshold for non-maximum suppression. |

## Methods

| Method | Summary |
|:-----|:--------|
| `DetectInstances(Tensor<>)` | Detects instances and returns their masks, bounding boxes, and class labels. |

