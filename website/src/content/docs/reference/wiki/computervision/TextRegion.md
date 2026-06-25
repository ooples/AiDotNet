---
title: "TextRegion<T>"
description: "Represents a detected text region in an image."
section: "API Reference"
---

`Models & Types` · `AiDotNet.ComputerVision.Detection.TextDetection`

Represents a detected text region in an image.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `TextRegion(BoundingBox<>,)` | Creates a new text region. |

## Properties

| Property | Summary |
|:-----|:--------|
| `Box` | Bounding box around the text region. |
| `Confidence` | Confidence score of the detection. |
| `Polygon` | Polygon points defining the exact boundary (for rotated or curved text). |
| `RegionType` | Whether this region is likely a word vs a text line. |
| `RotationAngle` | Rotation angle of the text region in degrees (if applicable). |

## Methods

| Method | Summary |
|:-----|:--------|
| `FromPolygon(List<ValueTuple<,>>,)` | Creates a text region from polygon points. |

