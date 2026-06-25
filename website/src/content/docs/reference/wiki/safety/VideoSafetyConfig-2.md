---
title: "VideoSafetyConfig"
description: "Configuration for video safety modules."
section: "API Reference"
---

`Options & Configuration` · `AiDotNet.Safety`

Configuration for video safety modules.

## For Beginners

These settings control how video content is checked for safety.
Content moderation checks sampled frames for harmful imagery, while deepfake detection
analyzes temporal consistency to spot manipulated videos.

## Properties

| Property | Summary |
|:-----|:--------|
| `ContentModeration` | Gets or sets whether general video content moderation is enabled. |
| `DeepfakeDetection` | Gets or sets whether video deepfake detection is enabled. |
| `FrameSamplingRate` | Gets or sets the frame sampling rate (frames per second to check). |

