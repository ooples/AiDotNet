---
title: "SmallestMaxSize<T>"
description: "Resizes the image so that the shortest edge equals the specified max size, preserving aspect ratio."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Augmentation.Image`

Resizes the image so that the shortest edge equals the specified max size, preserving aspect ratio.

## For Beginners

If you have a 400x200 image and set max_size to 256,
the short side (200) gets scaled to 256, and the long side scales proportionally
to 512, giving you a 512x256 image. This is typically followed by a center crop.

## How It Works

SmallestMaxSize scales the image so the shorter dimension matches the target size,
with the longer dimension scaled proportionally. This is the standard preprocessing
for ImageNet evaluation: resize shortest side to 256, then center crop to 224.

**When to use:**

- ImageNet evaluation preprocessing (resize 256 → center crop 224)
- When you want all images to have at least a minimum dimension
- Before cropping operations to ensure sufficient image size

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `SmallestMaxSize(Int32,InterpolationMode,Double)` | Creates a new SmallestMaxSize augmentation. |

## Properties

| Property | Summary |
|:-----|:--------|
| `Interpolation` | Gets the interpolation mode. |
| `MaxSize` | Gets the target size for the shortest edge. |

## Methods

| Method | Summary |
|:-----|:--------|
| `ApplyAugmentation(ImageTensor<>,AugmentationContext<>)` | Resizes so the shortest edge equals MaxSize. |
| `GetParameters` |  |

