---
title: "SegmentInfo<T>"
description: "Metadata for a single segment in panoptic/instance output."
section: "API Reference"
---

`Models & Types` · `AiDotNet.ComputerVision.Segmentation.Common`

Metadata for a single segment in panoptic/instance output.

## Properties

| Property | Summary |
|:-----|:--------|
| `Area` | Area of this segment in pixels. |
| `BoundingBox` | Bounding box [x1, y1, x2, y2] in pixel coordinates (for thing segments). |
| `Centroid` | Centroid (x, y) of the segment. |
| `ClassId` | Class ID of this segment. |
| `ClassName` | Class name (if available). |
| `Id` | Unique segment ID. |
| `IsThing` | Whether this is a "thing" (true) or "stuff" (false) segment. |
| `Score` | Confidence score in [0, 1]. |

