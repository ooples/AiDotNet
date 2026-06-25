---
title: "VideoSegmentationOutput<T>"
description: "Output for video segmentation containing per-frame masks with temporal tracking."
section: "API Reference"
---

`Models & Types` · `AiDotNet.ComputerVision.Segmentation.Common`

Output for video segmentation containing per-frame masks with temporal tracking.

## For Beginners

Video segmentation tracks objects across frames. This output
contains masks for each frame along with tracking IDs that let you know which
object in frame 1 corresponds to which object in frame 50.

## Properties

| Property | Summary |
|:-----|:--------|
| `AverageFps` | Average frames per second achieved. |
| `Frames` | Per-frame segmentation results. |
| `TotalFrames` | Total number of frames processed. |
| `TotalInferenceTime` | Total inference time for all frames. |
| `TotalTrackedObjects` | Total number of tracked objects across all frames. |
| `TrackSummaries` | Per-object tracking summaries across the video. |

