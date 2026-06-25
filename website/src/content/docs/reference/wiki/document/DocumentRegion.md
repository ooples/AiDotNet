---
title: "DocumentRegion<T>"
description: "Represents a detected document region."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Document`

Represents a detected document region.

## Properties

| Property | Summary |
|:-----|:--------|
| `BoundingBox` | Gets the bounding box as [x1, y1, x2, y2]. |
| `Confidence` | Gets the detection confidence score (0-1). |
| `ConfidenceValue` | Gets the confidence as a double value. |
| `Index` | Gets the region index. |
| `InstanceMask` | Gets the instance segmentation mask for this region (if available). |
| `PolygonPoints` | Gets the polygon points for non-rectangular regions. |
| `ReadingOrderPosition` | Gets the reading order position (lower = earlier in reading order). |
| `RegionType` | Gets the type of document region. |

