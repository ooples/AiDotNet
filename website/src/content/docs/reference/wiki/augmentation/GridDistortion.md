---
title: "GridDistortion<T>"
description: "Applies grid-based distortion to an image."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Augmentation.Image`

Applies grid-based distortion to an image.

## For Beginners

Imagine overlaying a flexible grid on your image, then randomly
nudging each grid intersection point. The image warps smoothly between the displaced points,
creating natural-looking distortions.

## How It Works

Grid distortion divides the image into a grid of cells and randomly displaces the grid
intersection points, then interpolates the displacement across each cell. This creates
smooth, locally varying distortions.

**When to use:**

- OCR and document recognition
- Medical image augmentation
- Any task requiring smooth local distortions

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `GridDistortion(Int32,Double,Double,Double)` | Creates a new grid distortion augmentation. |

## Properties

| Property | Summary |
|:-----|:--------|
| `DistortLimit` | Gets the maximum distortion magnitude. |
| `FillValue` | Gets the fill value for out-of-bounds pixels. |
| `NumSteps` | Gets the number of grid steps in each dimension. |

## Methods

| Method | Summary |
|:-----|:--------|
| `ApplyAugmentation(ImageTensor<>,AugmentationContext<>)` | Applies the grid distortion. |
| `GetParameters` |  |

