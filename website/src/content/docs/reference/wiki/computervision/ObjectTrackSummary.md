---
title: "ObjectTrackSummary"
description: "Summary of an object's track across a video sequence."
section: "API Reference"
---

`Models & Types` · `AiDotNet.ComputerVision.Segmentation.Common`

Summary of an object's track across a video sequence.

## Properties

| Property | Summary |
|:-----|:--------|
| `AverageArea` | Average mask area across all visible frames (in pixels). |
| `AverageScore` | Average confidence score across all visible frames. |
| `ClassId` | Class ID of the tracked object. |
| `ClassName` | Class name (if available). |
| `FirstFrame` | First frame where this object appears. |
| `IsActive` | Whether this object is currently being tracked. |
| `LastFrame` | Last frame where this object appears. |
| `TrackingId` | Unique tracking ID for this object. |
| `VisibleFrameCount` | Total number of frames where this object is visible. |

