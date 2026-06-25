---
title: "PadToSquare<T>"
description: "Pads an image to make it square while preserving the original content centered."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Augmentation.Image`

Pads an image to make it square while preserving the original content centered.

## For Beginners

If your image is 200x300 pixels (tall rectangle), this will add
50 pixels of padding to the left and right, making it 300x300 (square). The original image
stays in the center.

## How It Works

PadToSquare adds padding to the shorter dimension of an image so that height equals width.
The original image is centered within the square, with padding distributed evenly on both sides.
This is useful when models require square inputs but you want to preserve aspect ratio.

**When to use:**

- When your model requires square input but images have varying aspect ratios
- When you want to avoid distortion from non-uniform resizing
- Object detection where distortion could change bounding box proportions

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `PadToSquare(PaddingMode,Double,Double)` | Creates a new PadToSquare augmentation. |

## Properties

| Property | Summary |
|:-----|:--------|
| `FillValue` | Gets the constant fill value (used when Mode is Constant). |
| `Mode` | Gets the padding mode. |

## Methods

| Method | Summary |
|:-----|:--------|
| `ApplyAugmentation(ImageTensor<>,AugmentationContext<>)` | Pads the image to a square. |
| `GetParameters` |  |

