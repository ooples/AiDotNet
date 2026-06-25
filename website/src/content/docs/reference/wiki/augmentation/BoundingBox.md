---
title: "BoundingBox<T>"
description: "Represents a bounding box annotation for object detection."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Augmentation.Image`

Represents a bounding box annotation for object detection.

## For Beginners

A bounding box is simply a rectangle around an object
in an image, defined by its coordinates. Different systems use different ways
to specify these coordinates (corners vs center, pixels vs percentages).

## How It Works

Bounding boxes are rectangular regions that localize objects in images.
This class supports multiple coordinate formats used by different frameworks
and can convert between them.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `BoundingBox` | Creates an empty bounding box. |
| `BoundingBox(,,,,BoundingBoxFormat,Int32)` | Creates a bounding box with the specified coordinates. |

## Properties

| Property | Summary |
|:-----|:--------|
| `ClassIndex` | Gets or sets the class label index. |
| `ClassName` | Gets or sets the class label name. |
| `Confidence` | Gets or sets the confidence score (if from a detector). |
| `Format` | Gets or sets the current coordinate format. |
| `ImageHeight` | Gets or sets the image height (needed for normalized formats). |
| `ImageWidth` | Gets or sets the image width (needed for normalized formats). |
| `Metadata` | Gets or sets additional metadata. |
| `X1` | Gets or sets the X coordinate of the first point. |
| `X2` | Gets or sets the X coordinate of the second point or width. |
| `Y1` | Gets or sets the Y coordinate of the first point. |
| `Y2` | Gets or sets the Y coordinate of the second point or height. |

## Methods

| Method | Summary |
|:-----|:--------|
| `Area` | Calculates the area of this bounding box. |
| `Clip(Int32,Int32)` | Clips this bounding box to image boundaries. |
| `Clone` | Creates a deep copy of this bounding box. |
| `IoU(BoundingBox<>)` | Calculates the IoU (Intersection over Union) with another bounding box. |
| `IsValid` | Checks if this bounding box is valid (has positive area). |
| `ToCXCYWH` | Converts this bounding box to CXCYWH format. |
| `ToXYWH` | Converts this bounding box to XYWH format. |
| `ToXYXY` | Converts this bounding box to XYXY format. |
| `ToYOLO` | Converts this bounding box to YOLO format (normalized). |

