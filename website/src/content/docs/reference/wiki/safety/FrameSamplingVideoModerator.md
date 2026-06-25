---
title: "FrameSamplingVideoModerator<T>"
description: "Video content moderator that samples frames and applies image safety classification."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Safety.Video`

Video content moderator that samples frames and applies image safety classification.

## For Beginners

Videos are just sequences of images (frames). This module picks
some of those frames at regular intervals and checks each one for harmful content.
If any sampled frame is flagged, the entire video is flagged.

## How It Works

This module samples video frames at a configurable rate and applies image-level safety
checks (NSFW, violence) to each sampled frame. This is the standard approach used by
production video moderation systems: rather than processing every frame, a subset is
selected and each is independently classified.

**Sampling strategy:**

- At 1 FPS sampling on a 30 FPS video, only ~3% of frames are analyzed
- Higher sampling rates improve detection but increase processing time

**References:**

- YouTube content moderation: frame-level classification pipeline (2024)
- Efficient video understanding via sampling strategies (CVPR 2024)
- Video content moderation at scale (Meta, 2024)

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `FrameSamplingVideoModerator(Double,Double,Double)` | Initializes a new frame-sampling video moderator. |

## Properties

| Property | Summary |
|:-----|:--------|
| `ModuleName` |  |

## Methods

| Method | Summary |
|:-----|:--------|
| `EvaluateVideo(IReadOnlyList<Tensor<>>,Double)` |  |

