---
title: "FastDVDNet<T>"
description: "FastDVDNet: Towards Real-Time Deep Video Denoising Without Flow Estimation."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Video.Denoising`

FastDVDNet: Towards Real-Time Deep Video Denoising Without Flow Estimation.

## For Beginners

FastDVDNet removes noise from video while preserving details
and maintaining temporal consistency across frames. Unlike image denoisers,
it uses multiple frames to reduce noise more effectively.

Key advantages:

- Real-time video denoising
- No optical flow computation needed
- Handles various noise levels
- Preserves temporal consistency

Example usage:

## How It Works

**Technical Details:**

- Two-stage denoising pipeline
- Stage 1: Denoise groups of 3 frames
- Stage 2: Fuse stage 1 outputs temporally
- Noise map as additional input for noise-level adaptation

**Reference:** "FastDVDnet: Towards Real-Time Deep Video Denoising Without Flow Estimation"
https://arxiv.org/abs/1907.01361

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `FastDVDNet` | Initializes a new instance with default architecture settings. |

## Methods

| Method | Summary |
|:-----|:--------|
| `AddGaussianNoise(Tensor<>,Double)` | Adds synthetic noise to frames for testing. |
| `Denoise(List<Tensor<>>,Double)` | Denoises a sequence of video frames. |
| `Denoise(Tensor<>)` |  |
| `DenoiseFrame(List<Tensor<>>,Double)` | Denoises a single frame using neighboring frames. |
| `EstimateNoiseLevel(Tensor<>)` | Estimates the noise level in a frame. |
| `GetOptions` |  |
| `PostprocessOutput(Tensor<>)` |  |
| `PreprocessFrames(Tensor<>)` |  |

