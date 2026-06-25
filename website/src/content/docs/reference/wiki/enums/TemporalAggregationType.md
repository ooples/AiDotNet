---
title: "TemporalAggregationType"
description: "Specifies how frame-level features are aggregated into a single video-level representation."
section: "API Reference"
---

`Enums` · `AiDotNet.Enums`

Specifies how frame-level features are aggregated into a single video-level representation.

## For Beginners

When processing a video, each frame produces its own features.
This enum controls how those per-frame features are combined into one representation
for the whole video.

## Fields

| Field | Summary |
|:-----|:--------|
| `LastFrame` | Takes only the last frame's features as the video representation. |
| `MeanPooling` | Simple average (mean pooling) across all frame features. |
| `TemporalTransformer` | Uses a temporal transformer to learn attention-weighted aggregation across frames. |

