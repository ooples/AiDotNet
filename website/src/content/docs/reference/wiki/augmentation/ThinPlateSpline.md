---
title: "ThinPlateSpline<T>"
description: "Applies thin plate spline (TPS) transformation to an image."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Augmentation.Image`

Applies thin plate spline (TPS) transformation to an image.

## For Beginners

Imagine pinning a flexible sheet at several points and then
moving some pins. The sheet bends smoothly between the pins. TPS creates the smoothest
possible distortion that matches all the specified control point movements.

## How It Works

Thin plate spline is a smooth interpolation method that warps an image by specifying
control point displacements. TPS minimizes the bending energy of the deformation,
producing the smoothest possible warp that passes through all control points.
Named after the physical analogy of bending a thin metal plate.

**When to use:**

- Shape deformation augmentation
- Medical image registration
- Face warping and morphing

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `ThinPlateSpline(Int32,Double,Double,Double)` | Creates a new thin plate spline augmentation. |

## Properties

| Property | Summary |
|:-----|:--------|
| `FillValue` | Gets the fill value for out-of-bounds pixels. |
| `NumControlPoints` | Gets the number of control points per dimension. |
| `Scale` | Gets the maximum displacement scale. |

## Methods

| Method | Summary |
|:-----|:--------|
| `ApplyAugmentation(ImageTensor<>,AugmentationContext<>)` | Applies TPS transformation by computing per-pixel warp from control point displacements. |
| `GetParameters` |  |

