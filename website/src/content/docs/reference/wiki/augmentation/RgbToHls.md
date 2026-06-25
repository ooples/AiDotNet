---
title: "RgbToHls<T>"
description: "Converts an image between RGB and HLS (Hue, Lightness, Saturation) color spaces."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Augmentation.Image`

Converts an image between RGB and HLS (Hue, Lightness, Saturation) color spaces.

## For Beginners

HLS is similar to HSV but uses "lightness" instead of "value".
Lightness = 0 is always black, lightness = 1 is always white, and lightness = 0.5 gives
you the purest colors. This matches how we naturally think about light and dark colors.

## How It Works

HLS (also called HSL) separates color into hue, lightness, and saturation components.
Unlike HSV where value represents the maximum channel, lightness represents the average
of the maximum and minimum channels, making it more perceptually uniform.

**Channel layout:**

- **H (Hue)**: Color angle, normalized to [0, 1]
- **L (Lightness)**: [0, 1] (0 = black, 0.5 = pure color, 1 = white)
- **S (Saturation)**: [0, 1] (0 = gray, 1 = fully saturated)

**When to use:**

- Color manipulation where lightness should be independent of saturation
- CSS-style color adjustments

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `RgbToHls(Double)` | Creates a new RGB to HLS conversion. |

## Methods

| Method | Summary |
|:-----|:--------|
| `ApplyAugmentation(ImageTensor<>,AugmentationContext<>)` | Converts the image from RGB to HLS color space. |
| `GetParameters` |  |
| `HlsToRgb(Double,Double,Double)` | Converts HLS values back to RGB. |

