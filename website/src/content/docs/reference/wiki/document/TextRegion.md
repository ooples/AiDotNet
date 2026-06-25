---
title: "TextRegion<T>"
description: "Represents a detected text region in an image."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Document`

Represents a detected text region in an image.

## Properties

| Property | Summary |
|:-----|:--------|
| `BoundingBox` | Gets the bounding box as [x1, y1, x2, y2] for axis-aligned rectangles. |
| `Confidence` | Gets the detection confidence score (0-1). |
| `ConfidenceValue` | Gets the confidence as a double value. |
| `CroppedImage` | Gets the cropped image of this text region (if available). |
| `Index` | Gets the region index. |
| `PolygonPoints` | Gets the polygon points for rotated/curved text (list of [x, y] coordinates). |
| `RotationAngle` | Gets the rotation angle in degrees (if detected). |

