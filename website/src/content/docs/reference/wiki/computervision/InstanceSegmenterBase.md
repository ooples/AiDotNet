---
title: "InstanceSegmenterBase<T>"
description: "Base class for instance segmentation models."
section: "API Reference"
---

`Base Classes` · `AiDotNet.ComputerVision.Segmentation.InstanceSegmentation`

Base class for instance segmentation models.

## For Beginners

This base class provides the common functionality for all instance segmentation models, which detect and mask individual objects in images.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `InstanceSegmenterBase(InstanceSegmentationOptions<>)` | Creates a new instance segmenter. |

## Properties

| Property | Summary |
|:-----|:--------|
| `Name` | Name of this segmentation model. |

## Methods

| Method | Summary |
|:-----|:--------|
| `ApplyMaskNMS(List<InstanceMask<>>,Double)` | Applies NMS to filter overlapping instances. |
| `ApplySigmoid(Tensor<>)` | Applies sigmoid activation to tensor. |
| `BinarizeMask(Tensor<>,)` | Binarizes a mask using threshold. |
| `CropMaskToBox(Tensor<>,BoundingBox<>,Int32,Int32)` | Crops a mask to bounding box region. |
| `GetParameterCount` | Gets the total parameter count. |
| `LoadWeightsAsync(String,CancellationToken)` | Loads pretrained weights. |
| `ResizeMask(Tensor<>,Int32,Int32)` | Resizes a mask to target size using bilinear interpolation. |
| `SaveWeights(String)` | Saves model weights. |
| `Segment(Tensor<>)` | Performs instance segmentation on an image. |

