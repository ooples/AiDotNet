---
title: "Resize<T>"
description: "Resizes an image to a target size using configurable interpolation."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Augmentation.Image`

Resizes an image to a target size using configurable interpolation.

## For Beginners

Resizing changes the pixel dimensions of an image. When making
an image smaller, pixels must be combined. When making it larger, new pixels must be
created by interpolating between existing ones. The interpolation mode controls how
this is done.

## How It Works

Resize changes the spatial dimensions of an image using various interpolation methods.
Different interpolation modes trade off between speed and quality:

- **Nearest**: Fastest, produces blocky results. Best for masks/labels.
- **Bilinear**: Good balance of speed and quality. Default for most tasks.
- **Bicubic**: Smoother results, slower. Good for high-quality resizing.
- **Lanczos**: Highest quality, slowest. Best for final output.
- **Area**: Best for downscaling, averages pixel areas.

**When to use:**

- Preparing images for model input (most models require fixed input size)
- Downscaling large images to reduce memory and computation
- Upscaling small images when needed

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `Resize(Int32,Int32,InterpolationMode,Double)` | Creates a new resize augmentation. |

## Properties

| Property | Summary |
|:-----|:--------|
| `Interpolation` | Gets the interpolation mode used for resizing. |
| `TargetHeight` | Gets the target height. |
| `TargetWidth` | Gets the target width. |

## Methods

| Method | Summary |
|:-----|:--------|
| `ApplyAugmentation(ImageTensor<>,AugmentationContext<>)` | Applies the resize operation to the image. |
| `GetParameters` |  |

