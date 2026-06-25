---
title: "FlowFormer<T>"
description: "FlowFormer: A Transformer Architecture for Optical Flow."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Video.Motion`

FlowFormer: A Transformer Architecture for Optical Flow.

## For Beginners

FlowFormer estimates optical flow - the apparent motion of objects
between consecutive video frames. Unlike traditional methods, it uses transformers
to capture long-range dependencies in the cost volume.

Optical flow is useful for:

- Video stabilization
- Object tracking
- Action recognition
- Video editing and effects

Example usage:

## How It Works

**Technical Details:**

- Transformer-based cost volume aggregation
- Latent cost tokens for efficient memory
- Iterative flow refinement
- State-of-the-art accuracy on Sintel and KITTI benchmarks

**Reference:** "FlowFormer: A Transformer Architecture for Optical Flow" ECCV 2022
https://arxiv.org/abs/2203.16194

## Methods

| Method | Summary |
|:-----|:--------|
| `EstimateBidirectionalFlow(Tensor<>,Tensor<>)` | Estimates bidirectional flow (forward and backward). |
| `EstimateFlow(Tensor<>,Tensor<>)` | Estimates optical flow between two frames. |
| `EstimateFlowForVideo(List<Tensor<>>)` | Computes flow for all consecutive frame pairs in a video. |
| `GetOptions` |  |
| `PostprocessOutput(Tensor<>)` |  |
| `PreprocessFrames(Tensor<>)` |  |
| `WarpWithFlow(Tensor<>,Tensor<>)` | Warps an image using the estimated flow. |

## Fields

| Field | Summary |
|:-----|:--------|
| `DefaultResolution` | Initializes a new instance with default architecture settings. |

