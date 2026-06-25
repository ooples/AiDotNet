---
title: "EDVR<T>"
description: "EDVR: Video Restoration with Enhanced Deformable Convolutional Networks."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Video.Enhancement`

EDVR: Video Restoration with Enhanced Deformable Convolutional Networks.

## For Beginners

EDVR is a state-of-the-art video restoration model that can:

- Upscale low-resolution video (super-resolution)
- Remove blur and noise (deblurring/denoising)
- Handle complex motions and occlusions

The model uses alignment to compensate for motion between frames,
then fuses information from multiple frames to restore high-quality output.

Example usage:

## How It Works

**Technical Details:**

- PCD (Pyramid, Cascading and Deformable) alignment module
- TSA (Temporal and Spatial Attention) fusion module
- Deformable convolutions for motion-adaptive feature alignment

**Reference:** "EDVR: Video Restoration with Enhanced Deformable Convolutional Networks" CVPR 2019
https://arxiv.org/abs/1905.02716

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `EDVR` | Initializes a new instance with default architecture settings. |

## Methods

| Method | Summary |
|:-----|:--------|
| `Enhance(List<Tensor<>>)` | Enhances video frames with super-resolution and restoration. |
| `EnhanceFrame(List<Tensor<>>)` | Enhances a single frame using neighboring frames. |
| `GetOptions` |  |
| `PostprocessOutput(Tensor<>)` |  |
| `PreprocessFrames(Tensor<>)` |  |
| `Upscale(Tensor<>)` |  |

