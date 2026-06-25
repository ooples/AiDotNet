---
title: "DIFRINT<T>"
description: "DIFRINT: Deep Iterative Frame Interpolation for Full-frame Video Stabilization."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Video.Stabilization`

DIFRINT: Deep Iterative Frame Interpolation for Full-frame Video Stabilization.

## For Beginners

DIFRINT stabilizes shaky video by generating smooth intermediate frames.
Unlike traditional stabilization that crops the frame, DIFRINT synthesizes full frames
without losing any content from the edges.

Key advantages:

- Full-frame stabilization (no cropping)
- Handles large camera motions
- Synthesizes missing content from warping
- Real-time performance possible

Example usage:

## How It Works

**Technical Details:**

- Iterative refinement of stabilized frames
- Flow-based motion estimation
- Content synthesis for occluded regions
- Temporal consistency enforcement

**Reference:** "DIFRINT: A Framework for Full-Frame Video Stabilization"
https://arxiv.org/abs/2005.07055

## Methods

| Method | Summary |
|:-----|:--------|
| `ComputeBlockSAD(Tensor<>,Tensor<>,Int32,Int32,Int32,Int32,Int32,Int32,Int32)` | Computes Sum of Absolute Differences (SAD) between two blocks. |
| `EstimateMotionBetweenFrames(Tensor<>,Tensor<>)` | Estimates motion between two frames using block matching with subpixel refinement. |
| `EstimateMotionPath(List<Tensor<>>)` | Estimates camera motion between frames. |
| `EstimateRotation(List<ValueTuple<Double,Double,Int32,Int32>>,Double,Double,Double,Double)` | Estimates rotation angle from motion vectors using least squares fitting. |
| `GetOptions` |  |
| `PostprocessOutput(Tensor<>)` |  |
| `PreprocessFrames(Tensor<>)` |  |
| `Stabilize(List<Tensor<>>)` | Stabilizes a sequence of video frames. |
| `Stabilize(Tensor<>)` |  |
| `StabilizeFrame(Tensor<>,Tensor<>,Tensor<>)` | Stabilizes a single frame using neighboring frames. |

## Fields

| Field | Summary |
|:-----|:--------|
| `DefaultResolution` | Initializes a new instance with default architecture settings. |

