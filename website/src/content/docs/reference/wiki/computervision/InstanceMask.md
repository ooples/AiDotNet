---
title: "InstanceMask<T>"
description: "Represents a single instance with bounding box and segmentation mask."
section: "API Reference"
---

`Models & Types` · `AiDotNet.ComputerVision.Segmentation.InstanceSegmentation`

Represents a single instance with bounding box and segmentation mask.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `InstanceMask(BoundingBox<>,Tensor<>,Int32,)` | Creates a new instance mask. |

## Properties

| Property | Summary |
|:-----|:--------|
| `Box` | Bounding box around the instance. |
| `ClassId` | Class ID of the detected instance. |
| `ClassName` | Class name (if available). |
| `Confidence` | Detection confidence score. |
| `Mask` | Binary segmentation mask for this instance [height, width]. |
| `MaskConfidence` | Mask confidence score (if separate from detection confidence). |

## Methods

| Method | Summary |
|:-----|:--------|
| `ComputeMaskIoU(InstanceMask<>,INumericOperations<>)` | Computes IoU with another instance mask. |
| `GetMaskArea(INumericOperations<>)` | Gets the mask area (number of positive pixels). |

