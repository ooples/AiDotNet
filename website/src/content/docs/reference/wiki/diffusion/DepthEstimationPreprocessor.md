---
title: "DepthEstimationPreprocessor<T>"
description: "Depth estimation preprocessor for ControlNet conditioning."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Diffusion.Preprocessing`

Depth estimation preprocessor for ControlNet conditioning.

## For Beginners

This estimates how far away each part of the image is.
Bright areas are close, dark areas are far away. ControlNet uses this to
generate images with correct 3D perspective and depth.

In production, models like MiDaS or Depth Anything would be used for accurate
depth estimation. This implementation provides a gradient-based approximation.

## How It Works

Estimates monocular depth from a single image using gradient-based approximation.
The output is a single-channel depth map where brighter values indicate closer objects.

Reference: Ranftl et al., "Towards Robust Monocular Depth Estimation", IEEE TPAMI 2022

## Properties

| Property | Summary |
|:-----|:--------|
| `OutputChannels` |  |
| `OutputControlType` |  |

## Methods

| Method | Summary |
|:-----|:--------|
| `Transform(Tensor<>)` |  |

