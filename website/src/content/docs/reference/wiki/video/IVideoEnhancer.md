---
title: "IVideoEnhancer<T>"
description: "Interface for video enhancement models that improve video quality."
section: "API Reference"
---

`Interfaces` · `AiDotNet.Video.Interfaces`

Interface for video enhancement models that improve video quality.

## For Beginners

Video enhancement is like photo editing, but for videos.
These models can:

- Make blurry videos sharper (super-resolution)
- Remove grain/noise from old or low-light videos
- Smooth out shaky camera footage (stabilization)
- Make choppy videos smoother (frame interpolation)

The enhanced video has the same content but looks much better!

Example:

## How It Works

Video enhancers take degraded or low-quality video and produce improved versions.
Common enhancement tasks include super-resolution (upscaling), denoising,
stabilization, and frame interpolation.

## Properties

| Property | Summary |
|:-----|:--------|
| `EnhancementType` | Gets the type of enhancement this model performs. |
| `ScaleFactor` | Gets the scale factor for spatial enhancement (e.g., 2x, 4x upscaling). |
| `SupportsRealTime` | Gets whether this enhancer can process video in real-time. |
| `TemporalScaleFactor` | Gets the temporal scale factor for frame rate enhancement. |

## Methods

| Method | Summary |
|:-----|:--------|
| `ComputePSNR(Tensor<>,Tensor<>)` | Computes the Peak Signal-to-Noise Ratio between original and enhanced video. |
| `ComputeSSIM(Tensor<>,Tensor<>)` | Computes the Structural Similarity Index between original and enhanced video. |
| `Enhance(Tensor<>)` | Enhances a video and returns the improved version. |
| `EnhanceAsync(Tensor<>,IProgress<EnhancementProgress>,CancellationToken)` | Enhances a video asynchronously with progress reporting. |
| `EnhanceFrame(Tensor<>)` | Enhances a single frame from a video. |
| `EnhanceWithSlidingWindow(Tensor<>,Int32,Int32)` | Enhances a video using a sliding window approach for memory efficiency. |

