---
title: "MultimodalVideoModerator<T>"
description: "Comprehensive video moderator that combines frame-level content classification, temporal deepfake detection, and optional audio track analysis."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Safety.Video`

Comprehensive video moderator that combines frame-level content classification,
temporal deepfake detection, and optional audio track analysis.

## For Beginners

This is the "all-in-one" video safety checker. It looks at individual
frames for harmful images, checks whether the video flows naturally between frames (deepfakes
often don't), and analyzes scene transitions to find where content might have been spliced in.

## How It Works

Orchestrates multiple detection strategies for complete video safety analysis:

1. Frame sampling with ensemble image classifiers (NSFW, violence, hate symbols)
2. Temporal consistency analysis for deepfake detection
3. Scene transition analysis for detecting spliced/manipulated segments
4. Motion analysis for detecting unnatural movement patterns

**References:**

- Efficient video understanding via multi-scale temporal sampling (CVPR 2024)
- Spatio-temporal consistency for video deepfake detection (2025)
- Video content moderation at scale (Meta, 2024)
- VideoGuard: Multimodal video safety with reasoning-based instruction hierarchy (2025)

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `MultimodalVideoModerator(Double,Double,Double,Double,Double)` | Initializes a new multimodal video moderator. |

## Properties

| Property | Summary |
|:-----|:--------|
| `ModuleName` |  |

## Methods

| Method | Summary |
|:-----|:--------|
| `EvaluateVideo(IReadOnlyList<Tensor<>>,Double)` |  |

