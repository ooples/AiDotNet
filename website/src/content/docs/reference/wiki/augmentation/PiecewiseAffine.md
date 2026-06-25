---
title: "PiecewiseAffine<T>"
description: "Applies piecewise affine transformation by dividing the image into triangular regions."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Augmentation.Image`

Applies piecewise affine transformation by dividing the image into triangular regions.

## For Beginners

This divides your image into a grid of triangles, then slightly
shifts each grid point. Each triangle is then stretched/warped independently but connects
smoothly to its neighbors, creating a natural-looking distortion.

## How It Works

Piecewise affine transformation places a regular grid over the image, randomly displaces
the grid points, triangulates the grid, then applies a separate affine transform within
each triangle. This creates smooth, locally-varying warps that are more controlled than
elastic deformation.

**When to use:**

- Character/handwriting recognition augmentation
- Face augmentation with controlled deformation
- When you need smoother distortion than elastic transform

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `PiecewiseAffine(Int32,Int32,Double,Double,Double)` | Creates a new piecewise affine augmentation. |

## Properties

| Property | Summary |
|:-----|:--------|
| `FillValue` | Gets the fill value for out-of-bounds pixels. |
| `GridCols` | Gets the number of grid columns. |
| `GridRows` | Gets the number of grid rows. |
| `Scale` | Gets the maximum displacement scale as fraction of grid cell size. |

## Methods

| Method | Summary |
|:-----|:--------|
| `ApplyAugmentation(ImageTensor<>,AugmentationContext<>)` | Applies the piecewise affine transformation using grid-based interpolation. |
| `GetParameters` |  |

