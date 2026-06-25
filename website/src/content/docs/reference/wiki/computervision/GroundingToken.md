---
title: "GroundingToken"
description: "A grounding token linking a text span to an image region."
section: "API Reference"
---

`Models & Types` · `AiDotNet.ComputerVision.Segmentation.Common`

A grounding token linking a text span to an image region.

## Properties

| Property | Summary |
|:-----|:--------|
| `BoundingBox` | Bounding box [x1, y1, x2, y2] of the grounded region. |
| `Confidence` | Confidence of the grounding. |
| `EndIndex` | End character index in the full text response. |
| `MaskIndex` | Mask index in the output masks tensor (if a mask was generated). |
| `StartIndex` | Start character index in the full text response. |
| `Text` | Text span that was grounded. |

