---
title: "AnchorGenerator<T>"
description: "Generates anchor boxes for object detection models."
section: "API Reference"
---

`Models & Types` · `AiDotNet.ComputerVision.Detection.Anchors`

Generates anchor boxes for object detection models.

## For Beginners

Anchor boxes (also called prior boxes) are pre-defined boxes
of various sizes and aspect ratios placed at each location in the feature map.
The detector predicts how to adjust these anchors to fit actual objects.
Using anchors helps the model handle objects of different sizes and shapes.

## How It Works

For example, a feature map of 80x80 with 3 anchors per location generates
80 x 80 x 3 = 19,200 anchor boxes, each representing a potential object location.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `AnchorGenerator` | Creates a new anchor generator with default YOLO-style anchors. |
| `AnchorGenerator(Double[],Double[],Double[],Int32[])` | Creates a new anchor generator with custom settings. |

## Properties

| Property | Summary |
|:-----|:--------|
| `AspectRatios` | Aspect ratios for anchors (height/width). |
| `BaseSizes` | Base sizes for anchors at each feature level. |
| `NumAnchorsPerLocation` | Gets the number of anchors per feature map location. |
| `Scales` | Scales to apply to base sizes. |
| `Strides` | Strides (downsampling factors) at each feature level. |

## Methods

| Method | Summary |
|:-----|:--------|
| `CreateFasterRCNNAnchors` | Creates an anchor generator with Faster R-CNN style anchors. |
| `CreateRetinaNetAnchors` | Creates an anchor generator with RetinaNet style anchors. |
| `CreateYOLOAnchors(Double[0:,0:],Int32[])` | Creates an anchor generator with YOLO-style anchors. |
| `GenerateAnchors(List<ValueTuple<Int32,Int32>>)` | Generates anchors for all feature levels. |
| `GenerateAnchorsForImage(Int32,Int32)` | Generates anchors for an image of specified size. |
| `GenerateAnchorsForLevel(Int32,Int32,Int32,Double)` | Generates anchors for a single feature map level. |
| `GenerateBaseAnchors(Double)` | Generates base anchors (centered at origin) for all aspect ratios and scales. |
| `GetTotalAnchorCount(Int32,Int32)` | Gets the total number of anchors for an image. |

