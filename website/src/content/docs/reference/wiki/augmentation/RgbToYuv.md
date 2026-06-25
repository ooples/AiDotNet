---
title: "RgbToYuv<T>"
description: "Converts an image between RGB and YUV color spaces."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Augmentation.Image`

Converts an image between RGB and YUV color spaces.

## For Beginners

YUV splits an image into brightness (Y) and color (U, V).
The Y channel alone looks like a grayscale version of the image. This is useful because
the human eye is more sensitive to brightness changes than color changes, so you can
compress the U and V channels more aggressively.

## How It Works

YUV separates luminance (Y) from chrominance (U and V). This encoding was originally
designed for analog television to allow backwards compatibility with black-and-white sets.
The Y channel carries all the brightness information, while U and V carry color.

**Channel layout (BT.601 standard):**

- **Y (Luminance)**: [0, 1] - Brightness
- **U (Cb)**: [-0.436, 0.436] - Blue difference
- **V (Cr)**: [-0.615, 0.615] - Red difference

**When to use:**

- Video processing and JPEG/MPEG encoding
- Separating luminance for brightness-invariant processing
- Chroma subsampling applications

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `RgbToYuv(Double)` | Creates a new RGB to YUV conversion. |

## Methods

| Method | Summary |
|:-----|:--------|
| `ApplyAugmentation(ImageTensor<>,AugmentationContext<>)` | Converts the image from RGB to YUV color space (BT.601 standard). |
| `GetParameters` |  |
| `YuvToRgb(Double,Double,Double)` | Converts YUV values back to RGB. |

