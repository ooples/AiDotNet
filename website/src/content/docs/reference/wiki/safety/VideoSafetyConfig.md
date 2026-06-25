---
title: "VideoSafetyConfig"
description: "Configuration for video safety detection modules."
section: "API Reference"
---

`Options & Configuration` · `AiDotNet.Safety.Video`

Configuration for video safety detection modules.

## For Beginners

Use this to configure video content moderation including
frame sampling rate, deepfake detection, and content classification settings.

## Properties

| Property | Summary |
|:-----|:--------|
| `FrameSamplingRate` | Frame sampling rate (frames per second to analyze). |
| `MaxFrames` | Maximum frames to analyze per video. |
| `ModerationThreshold` | Content moderation threshold (0.0-1.0). |
| `TemporalAnalysis` | Whether to use temporal consistency analysis for deepfake detection. |

