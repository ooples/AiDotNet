---
title: "IVideoSegmentation<T>"
description: "Interface for video segmentation models that track and segment objects across video frames."
section: "API Reference"
---

`Interfaces` · `AiDotNet.Interfaces`

Interface for video segmentation models that track and segment objects across video frames.

## For Beginners

Video segmentation is like image segmentation but for videos.
The key challenge is tracking: once you segment a car in frame 1, you need to keep
tracking that same car through all subsequent frames, even when it moves, gets
partially hidden, or changes appearance.

Types of video segmentation:

- Semi-supervised VOS: You provide masks on the first frame, model tracks through video
- Unsupervised VOS: Model automatically finds and tracks salient objects
- Video Instance Segmentation (VIS): Detect + segment + track all instances per frame
- Video Panoptic Segmentation (VPS): Panoptic segmentation with tracking

Models implementing this interface:

- SAM 2 (Meta, streaming memory architecture)
- Cutie (CVPR 2024, object-level memory)
- XMem (ECCV 2022, three-level memory)
- DEVA (ICCV 2023, decoupled propagation)
- EfficientTAM (lightweight, mobile-ready)
- UniVS (CVPR 2024, universal video segmentation)

## How It Works

Video segmentation extends image segmentation to temporal sequences by tracking objects
across frames. Models maintain memory of previously seen objects to ensure consistent
segmentation throughout the video.

## Properties

| Property | Summary |
|:-----|:--------|
| `MaxTrackedObjects` | Gets the maximum number of objects that can be tracked simultaneously. |
| `SupportsStreaming` | Gets whether the model supports streaming (frame-by-frame) processing. |

## Methods

| Method | Summary |
|:-----|:--------|
| `AddCorrection(Int32,Tensor<>)` | Adds a correction mask for an object at the current frame. |
| `InitializeTracking(Tensor<>,Tensor<>,Int32[])` | Initializes tracking with masks on the first frame. |
| `PropagateToFrame(Tensor<>)` | Propagates segmentation masks to the next frame. |
| `ResetTracking` | Resets the tracking state and memory. |

