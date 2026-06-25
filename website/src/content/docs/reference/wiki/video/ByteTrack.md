---
title: "ByteTrack<T>"
description: "ByteTrack: Multi-Object Tracking by Associating Every Detection Box."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Video.Tracking`

ByteTrack: Multi-Object Tracking by Associating Every Detection Box.

## For Beginners

ByteTrack is a simple yet powerful multi-object tracking method
that tracks all detection boxes, including low-confidence ones that other trackers ignore.

Key capabilities:

- Track multiple objects across video frames
- Handle occlusions and crowded scenes
- Associate detections between frames using motion prediction
- Maintain object IDs consistently over time

Example usage:

## How It Works

**Technical Details:**

- YOLOX-based detector backbone
- Kalman filter for motion prediction
- Two-stage association (high + low confidence boxes)
- IoU-based matching with Hungarian algorithm

**Reference:** "ByteTrack: Multi-Object Tracking by Associating Every Detection Box"
https://arxiv.org/abs/2110.06864

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `ByteTrack` | Initializes a new instance with default architecture settings. |

## Methods

| Method | Summary |
|:-----|:--------|
| `Detect(Tensor<>)` | Detects objects in a single frame. |
| `GetOptions` |  |
| `ResetTracks` | Resets tracking state for a new video. |
| `Track(List<Tensor<>>)` | Tracks objects across video frames. |
| `UpdateTracks(List<Detection<>>)` | Updates tracks with new detections. |

