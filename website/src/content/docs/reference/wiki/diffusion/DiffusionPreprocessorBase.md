---
title: "DiffusionPreprocessorBase<T>"
description: "Base class for diffusion model condition preprocessors that convert input images into control signals (edge maps, depth maps, pose skeletons, etc.)."
section: "API Reference"
---

`Base Classes` · `AiDotNet.Diffusion.Preprocessing`

Base class for diffusion model condition preprocessors that convert input images
into control signals (edge maps, depth maps, pose skeletons, etc.).

## For Beginners

These preprocessors are the "preparation step" before using
ControlNet. They extract specific features from your image:

- Edge detection: finds outlines and boundaries
- Depth estimation: estimates how far each pixel is
- Pose detection: finds body keypoints
- Segmentation: identifies object regions

The output becomes the "blueprint" that guides image generation.

## How It Works

Diffusion preprocessors transform input images into condition maps that guide
controlled generation. For example, a Canny edge preprocessor converts a photo
into an edge map that ControlNet uses to preserve structure.

## Properties

| Property | Summary |
|:-----|:--------|
| `ColumnIndices` |  |
| `IsFitted` |  |
| `OutputChannels` | Gets the number of output channels for the control signal. |
| `OutputControlType` | Gets the control type this preprocessor produces. |
| `SupportsInverseTransform` |  |

## Methods

| Method | Summary |
|:-----|:--------|
| `Clamp(,,)` | Clamps a value between min and max. |
| `Fit(Tensor<>)` |  |
| `FitTransform(Tensor<>)` |  |
| `GetFeatureNamesOut(String[])` |  |
| `InverseTransform(Tensor<>)` |  |
| `ToGrayscale(,,)` | Converts an RGB pixel to grayscale using luminance weights. |
| `Transform(Tensor<>)` |  |

## Fields

| Field | Summary |
|:-----|:--------|
| `NumOps` | Provides numeric operations for the specific type T. |

