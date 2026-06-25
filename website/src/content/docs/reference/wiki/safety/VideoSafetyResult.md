---
title: "VideoSafetyResult"
description: "Detailed result from video safety evaluation with per-frame annotations."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Safety.Video`

Detailed result from video safety evaluation with per-frame annotations.

## For Beginners

VideoSafetyResult provides AI safety functionality. Default values follow the original paper settings.

## Properties

| Property | Summary |
|:-----|:--------|
| `FrameScores` | Per-frame safety scores (index → score). |
| `FramesAnalyzed` | Number of frames analyzed. |
| `IsSafe` | Whether the video is safe overall. |
| `TemporalConsistency` | Temporal consistency score (0.0 = inconsistent/deepfake, 1.0 = natural). |
| `UnsafeFrames` | Number of frames flagged as unsafe. |

