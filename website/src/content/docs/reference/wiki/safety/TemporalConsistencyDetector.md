---
title: "TemporalConsistencyDetector<T>"
description: "Detects deepfake videos by analyzing temporal consistency between consecutive frames."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Safety.Video`

Detects deepfake videos by analyzing temporal consistency between consecutive frames.

## For Beginners

When someone creates a fake video (deepfake), the way objects and
faces change between frames is often slightly "off" compared to real video. This module
looks at how each frame differs from the previous one and flags videos where those
differences look unnatural.

## How It Works

Deepfake videos often exhibit subtle temporal inconsistencies that are not present in
authentic video. This module analyzes frame-to-frame differences to detect anomalies
in motion, color, and spatial coherence that are characteristic of AI-generated or
manipulated video content.

**Detection approach:**

1. Compute frame-to-frame difference statistics (mean absolute difference, variance)
2. Analyze temporal smoothness — real video has consistent motion patterns
3. Detect sudden discontinuities that may indicate frame manipulation
4. Check for periodic artifacts from frame-by-frame generation
5. Measure temporal jitter — variation in frame difference magnitudes

**References:**

- Spatio-temporal consistency for video deepfake detection (2025)
- NACO: Self-supervised natural consistency for face forgery detection (ECCV 2024)
- Generalizable deepfake detection across benchmarks (CVPR 2025)
- FakeFormer: Efficient vulnerability-driven face forgery detection (2024)

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `TemporalConsistencyDetector(Double)` | Initializes a new temporal consistency detector. |

## Properties

| Property | Summary |
|:-----|:--------|
| `ModuleName` |  |

## Methods

| Method | Summary |
|:-----|:--------|
| `ComputePeriodicity(Vector<>,Int32)` | Detects periodicity in frame difference signal via autocorrelation. |
| `EstimateDeepfakeScore(TemporalConsistencyDetector<>.TemporalFeatures)` | Estimates deepfake probability from temporal features. |
| `EvaluateVideo(IReadOnlyList<Tensor<>>,Double)` |  |

