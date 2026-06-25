---
title: "PanopticSegment<T>"
description: "A single segment in a panoptic segmentation result."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Interfaces`

A single segment in a panoptic segmentation result.

## Properties

| Property | Summary |
|:-----|:--------|
| `Area` | Area of this segment in pixels. |
| `ClassId` | Class ID of this segment. |
| `Confidence` | Confidence score in [0, 1]. |
| `IsThing` | Whether this segment is a "thing" (true) or "stuff" (false). |
| `Mask` | Binary mask for this segment [H, W]. |
| `SegmentId` | Unique segment ID. |

