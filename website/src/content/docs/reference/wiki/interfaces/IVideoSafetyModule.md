---
title: "IVideoSafetyModule<T>"
description: "Interface for safety modules that operate on video content."
section: "API Reference"
---

`Interfaces` · `AiDotNet.Interfaces`

Interface for safety modules that operate on video content.

## For Beginners

Video safety modules check video content for harmful material.
They can analyze individual frames (like image safety) and also detect temporal
inconsistencies that reveal deepfake manipulation — things like unnatural blinking,
facial warping between frames, or audio-visual mismatches.

## How It Works

Video safety modules analyze video frames and their temporal relationships for safety
risks such as harmful visual content, deepfake videos, and policy-violating material.

**References:**

- Spatio-temporal consistency exploitation for deepfake video detection (2025)
- NACO: Self-supervised natural consistency for face forgery detection (ECCV 2024)
- Generalizable deepfake detection across benchmarks (CVPR 2025)

## Methods

| Method | Summary |
|:-----|:--------|
| `EvaluateVideo(IReadOnlyList<Tensor<>>,Double)` | Evaluates the given video frames for safety and returns any findings. |

