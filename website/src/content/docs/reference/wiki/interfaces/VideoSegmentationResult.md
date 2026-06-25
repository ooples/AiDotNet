---
title: "VideoSegmentationResult<T>"
description: "Result of video segmentation for a single frame."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Interfaces`

Result of video segmentation for a single frame.

## Properties

| Property | Summary |
|:-----|:--------|
| `Confidences` | Confidence scores for each tracked object. |
| `FrameIndex` | Frame index in the video sequence. |
| `IsVisible` | Whether each object is considered visible (not fully occluded) in this frame. |
| `Masks` | Per-object masks [numObjects, H, W]. |
| `ObjectIds` | Object IDs corresponding to each mask. |

