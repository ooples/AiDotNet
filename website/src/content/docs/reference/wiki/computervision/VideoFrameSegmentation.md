---
title: "VideoFrameSegmentation<T>"
description: "Segmentation result for a single video frame."
section: "API Reference"
---

`Models & Types` · `AiDotNet.ComputerVision.Segmentation.Common`

Segmentation result for a single video frame.

## Properties

| Property | Summary |
|:-----|:--------|
| `FrameIndex` | Frame index in the video sequence. |
| `InferenceTime` | Inference time for this frame. |
| `Masks` | Per-object masks for this frame. |
| `SemanticMap` | Combined semantic map for this frame [H, W] (if applicable). |
| `Timestamp` | Timestamp of this frame in the video (seconds). |
| `VisibleObjectCount` | Number of visible objects in this frame. |

