---
title: "VideoStabilizationBase<T>"
description: "Base class for video stabilization models that remove camera shake from video sequences."
section: "API Reference"
---

`Base Classes` · `AiDotNet.Video`

Base class for video stabilization models that remove camera shake from video sequences.

## For Beginners

Video stabilization removes camera shake from handheld video footage.
It analyzes how the camera moved between frames and compensates by shifting/warping each
frame to create a smooth viewing experience. Some methods crop the edges (like smartphone
stabilization), while advanced neural methods can fill in the missing edges.

## How It Works

Video stabilization compensates for unwanted camera motion to produce smooth footage.
This base class provides:

- Camera trajectory estimation and smoothing
- Crop ratio management (stabilization typically requires cropping)
- Homography/affine transform estimation utilities
- Full-frame inpainting support for crop-free stabilization

Derived classes implement specific architectures like DIFRINT, DUT, FuSta, etc.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `VideoStabilizationBase(NeuralNetworkArchitecture<>,ILossFunction<>,Double)` | Initializes a new instance of the VideoStabilizationBase class. |

## Properties

| Property | Summary |
|:-----|:--------|
| `CropRatio` | Gets the crop ratio used for stabilization (fraction of frame that may be cropped). |
| `SmoothingWindowSize` | Gets the smoothing window size for trajectory smoothing. |
| `SupportsFullFrame` | Gets whether this model supports full-frame stabilization (no cropping). |

## Methods

| Method | Summary |
|:-----|:--------|
| `EstimateTrajectory(Tensor<>)` | Estimates the camera trajectory (per-frame transforms) from the input sequence. |
| `PredictCore(Tensor<>)` |  |
| `SmoothTrajectory(List<Tensor<>>)` | Smooths a camera trajectory using a moving average filter. |
| `Stabilize(Tensor<>)` | Stabilizes a sequence of video frames. |

