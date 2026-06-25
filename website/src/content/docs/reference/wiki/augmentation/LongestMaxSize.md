---
title: "LongestMaxSize<T>"
description: "Resizes the image so that the longest edge equals the specified max size, preserving aspect ratio."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Augmentation.Image`

Resizes the image so that the longest edge equals the specified max size, preserving aspect ratio.

## For Beginners

If you have a 400x200 image and set max_size to 300,
the long side (400) gets scaled to 300, and the short side scales proportionally
to 150, giving you a 300x150 image.

## How It Works

LongestMaxSize scales the image so the longer dimension matches the target size,
with the shorter dimension scaled proportionally. This ensures the image fits within
a max_size x max_size box without exceeding it.

**When to use:**

- Preprocessing for object detection (DETR, Faster R-CNN)
- When you want to limit maximum dimension without distortion
- Batch processing with variable-size images

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `LongestMaxSize(Int32,InterpolationMode,Double)` | Creates a new LongestMaxSize augmentation. |

## Properties

| Property | Summary |
|:-----|:--------|
| `Interpolation` | Gets the interpolation mode. |
| `MaxSize` | Gets the maximum size for the longest edge. |

## Methods

| Method | Summary |
|:-----|:--------|
| `ApplyAugmentation(ImageTensor<>,AugmentationContext<>)` | Resizes so the longest edge equals MaxSize. |
| `GetParameters` |  |

