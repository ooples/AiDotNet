---
title: "ResizeWithAspectRatio<T>"
description: "Resizes an image to fit within target dimensions while preserving the aspect ratio, then pads to reach the exact target size."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Augmentation.Image`

Resizes an image to fit within target dimensions while preserving the aspect ratio,
then pads to reach the exact target size.

## For Beginners

If you have a wide 400x200 image and want 300x300 output,
this first shrinks it to 300x150 (keeping proportions), then adds 75 pixels of padding
top and bottom to reach 300x300. The image looks correct, just with borders.

## How It Works

This combines resizing and padding in one step. First, the image is scaled so that it
fits within the target dimensions without exceeding them. Then, padding is added to
reach the exact target size. This ensures no distortion while producing a fixed-size output.

**When to use:**

- When model requires fixed input size but aspect ratio matters
- Object detection where distorted proportions would hurt accuracy
- YOLO-style preprocessing (letterbox resizing)

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `ResizeWithAspectRatio(Int32,Int32,InterpolationMode,PaddingMode,Double,Double)` | Creates a new ResizeWithAspectRatio augmentation. |
| `ResizeWithAspectRatio(Int32,InterpolationMode,PaddingMode,Double,Double)` | Creates a square ResizeWithAspectRatio augmentation. |

## Properties

| Property | Summary |
|:-----|:--------|
| `FillValue` | Gets the fill value for constant padding. |
| `Interpolation` | Gets the interpolation mode used for resizing. |
| `PadMode` | Gets the padding mode for the fill areas. |
| `TargetHeight` | Gets the target height. |
| `TargetWidth` | Gets the target width. |

## Methods

| Method | Summary |
|:-----|:--------|
| `ApplyAugmentation(ImageTensor<>,AugmentationContext<>)` | Resizes the image preserving aspect ratio, then pads to target size. |
| `GetParameters` |  |

