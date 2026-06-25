---
title: "VideoSuperResolutionBase<T>"
description: "Base class for video super-resolution models that upscale low-resolution video to higher resolution."
section: "API Reference"
---

`Base Classes` · `AiDotNet.Video`

Base class for video super-resolution models that upscale low-resolution video to higher resolution.

## For Beginners

Video super-resolution makes low-resolution video sharper and more
detailed. For example, it can upscale a 480p video to 4K quality. Unlike single-image
methods, video SR uses information from neighboring frames for better quality and
temporal consistency (no flickering between frames).

## How It Works

Video super-resolution extends image super-resolution by exploiting temporal information
across multiple frames. This base class provides:

- Scale factor management (2x, 4x, 8x upscaling)
- Tile-based inference for memory-efficient processing of high-resolution video
- Bicubic upsampling as fallback/initialization
- Temporal consistency utilities

Derived classes implement specific architectures like BasicVSR++, RVRT, RealBasicVSR, etc.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `VideoSuperResolutionBase(NeuralNetworkArchitecture<>,ILossFunction<>,Double)` | Initializes a new instance of the VideoSuperResolutionBase class. |

## Properties

| Property | Summary |
|:-----|:--------|
| `ScaleFactor` | Gets the spatial upscaling factor (e.g., 2 for 2x, 4 for 4x). |
| `TileOverlap` | Gets or sets the overlap between adjacent tiles. |
| `TileSize` | Gets or sets the tile size for memory-efficient tiled processing. |

## Methods

| Method | Summary |
|:-----|:--------|
| `BilinearUpsample(Tensor<>,Int32)` | Performs bilinear upsampling as a baseline or initialization. |
| `EstimateFlow(Tensor<>,Tensor<>)` | Estimates optical flow between two frames for temporal alignment. |
| `PredictCore(Tensor<>)` |  |
| `Upscale(Tensor<>)` | Upscales a sequence of video frames. |

