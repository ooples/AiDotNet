---
title: "ReferringSegmentationResult<T>"
description: "Result of referring segmentation."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Interfaces`

Result of referring segmentation.

## Properties

| Property | Summary |
|:-----|:--------|
| `BoundingBoxes` | Bounding boxes for each referred object [numObjects, 4] as (x1, y1, x2, y2). |
| `Confidence` | Confidence score for the segmentation. |
| `FrameIndex` | Frame index (for video segmentation results). |
| `Masks` | Binary mask(s) for the referred object(s) [numObjects, H, W]. |
| `TextResponse` | The model's textual response explaining what was segmented. |

