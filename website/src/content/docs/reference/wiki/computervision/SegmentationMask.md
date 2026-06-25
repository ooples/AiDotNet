---
title: "SegmentationMask<T>"
description: "Represents a single segmentation mask with associated metadata."
section: "API Reference"
---

`Models & Types` · `AiDotNet.ComputerVision.Segmentation.Common`

Represents a single segmentation mask with associated metadata.

## For Beginners

A segmentation mask is a binary image where each pixel is either
part of the object (1) or not (0). This class wraps a mask with useful metadata like
the object's class, confidence score, area, and bounding box.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `SegmentationMask(Tensor<>,Int32,Double)` | Creates a new segmentation mask. |

## Properties

| Property | Summary |
|:-----|:--------|
| `Area` | Area of the mask in pixels. |
| `BoundingBox` | Bounding box [x1, y1, x2, y2] in pixel coordinates. |
| `Centroid` | Centroid (x, y) of the mask. |
| `ClassId` | Class ID of the segmented object. |
| `ClassName` | Class name (if available). |
| `InstanceId` | Instance ID (unique per instance for instance/panoptic segmentation). |
| `IsThing` | Whether this mask represents a "thing" (countable object) or "stuff" (amorphous region). |
| `Mask` | Binary mask tensor [H, W] where values > 0 indicate the segmented region. |
| `PredictedIoU` | IoU prediction score (if the model supports it). |
| `Score` | Confidence score in [0, 1]. |
| `StabilityScore` | Stability score measuring mask consistency under perturbations (SAM-style). |
| `TrackingId` | Object tracking ID (for video segmentation across frames). |

