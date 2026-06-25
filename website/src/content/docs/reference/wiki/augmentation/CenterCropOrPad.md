---
title: "CenterCropOrPad<T>"
description: "Crops or pads an image to reach the target size, centering the content."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Augmentation.Image`

Crops or pads an image to reach the target size, centering the content.

## For Beginners

This is a "smart resize" that doesn't stretch your image.
If the image is too big, it cuts away the edges. If too small, it adds borders. Either
way, the original content stays centered in the output.

## How It Works

CenterCropOrPad handles both cases: if the image is larger than the target, it center crops;
if smaller, it center pads with the specified fill value. This ensures a fixed output size
regardless of input dimensions.

**When to use:**

- When input images vary in size and you need fixed dimensions
- When you want to avoid any scaling/interpolation artifacts
- Evaluation/inference preprocessing

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `CenterCropOrPad(Int32,Double,Double)` | Creates a square CenterCropOrPad. |
| `CenterCropOrPad(Int32,Int32,Double,Double)` | Creates a new CenterCropOrPad augmentation. |

## Properties

| Property | Summary |
|:-----|:--------|
| `FillValue` | Gets the fill value for padding (when image is smaller than target). |
| `TargetHeight` | Gets the target height. |
| `TargetWidth` | Gets the target width. |

## Methods

| Method | Summary |
|:-----|:--------|
| `ApplyAugmentation(ImageTensor<>,AugmentationContext<>)` | Applies center crop or pad to reach target size. |
| `GetParameters` |  |

