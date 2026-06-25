---
title: "TemporalConsistency<T>"
description: "Temporal Consistency metric for evaluating video smoothness and coherence."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Metrics`

Temporal Consistency metric for evaluating video smoothness and coherence.

## How It Works

Temporal consistency measures how smoothly content changes between consecutive frames.
It's crucial for video generation quality, as flickering and jittery artifacts
are often more distracting than per-frame quality issues.

Two main approaches:

- Pixel-level temporal difference (simple but sensitive to motion)
- Optical flow consistency (accounts for motion but more complex)

Values range from 0 to 1, where higher values indicate better temporal consistency.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `TemporalConsistency` | Initializes a new instance of TemporalConsistency calculator. |

## Methods

| Method | Summary |
|:-----|:--------|
| `BilinearSample(Tensor<>,Double,Double,Int32,Int32,Int32)` | Performs bilinear sampling from a frame. |
| `ComputeFlicker(Tensor<>)` | Computes flicker metric (measures high-frequency temporal variations). |
| `ComputeFrameFlicker(Tensor<>,Tensor<>,Tensor<>)` | Computes flicker for a single frame based on its neighbors. |
| `ComputeSimple(Tensor<>)` | Computes simple temporal consistency without optical flow. |
| `ComputeWithFlow(Tensor<>,Tensor<>)` | Computes temporal consistency using warped frame differences. |
| `WarpFrame(Tensor<>,Int32,Tensor<>,Int32,Int32,Int32,Int32)` | Warps a frame using optical flow. |

