---
title: "RgbToXyz<T>"
description: "Converts an image between RGB and CIE XYZ color space."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Augmentation.Image`

Converts an image between RGB and CIE XYZ color space.

## For Beginners

XYZ is a "master" color space that represents all colors
the human eye can see. Other color spaces (like RGB, LAB) are derived from XYZ.
Y represents luminance (brightness), while X and Z represent chromaticity.
You typically don't use XYZ directly for augmentation, but it's needed as an
intermediate step for conversions to LAB and other spaces.

## How It Works

CIE XYZ is a device-independent color space that serves as the foundation for most
other color space conversions. It was defined by the International Commission on
Illumination (CIE) in 1931 based on human color perception experiments.

**Channel layout (D65 illuminant):**

- **X**: Mix of cone responses, roughly correlates with red
- **Y**: Luminance (brightness as perceived by human eye)
- **Z**: Roughly correlates with blue

**When to use:**

- As intermediate step for color space conversions (RGB → XYZ → LAB)
- Color science calculations and colorimetry
- White point adaptation

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `RgbToXyz(Double)` | Creates a new RGB to XYZ conversion. |

## Methods

| Method | Summary |
|:-----|:--------|
| `ApplyAugmentation(ImageTensor<>,AugmentationContext<>)` | Converts the image from sRGB to CIE XYZ color space using D65 illuminant. |
| `GetParameters` |  |
| `XyzToRgb(Double,Double,Double)` | Converts XYZ values back to sRGB. |

