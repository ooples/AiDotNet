---
title: "RgbToHsv<T>"
description: "Converts an image between RGB and HSV color spaces."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Augmentation.Image`

Converts an image between RGB and HSV color spaces.

## For Beginners

RGB describes colors by mixing red, green, and blue light.
HSV describes colors by their shade (hue, 0-360 degrees), how vivid they are
(saturation, 0-1), and how bright they are (value, 0-1). HSV is more intuitive for
color manipulation.

## How It Works

HSV (Hue, Saturation, Value) separates color information (hue) from intensity (value)
and purity (saturation). This makes it easier to perform color-based operations like
adjusting hue or saturation independently.

**Channel layout:**

- **H (Hue)**: Color angle, normalized to [0, 1] (representing 0-360 degrees)
- **S (Saturation)**: Color purity, [0, 1] (0 = gray, 1 = pure color)
- **V (Value)**: Brightness, [0, 1] (0 = black, 1 = brightest)

**When to use:**

- Color-based augmentation (adjusting hue/saturation independently)
- Color-based object detection or segmentation
- When you need to separate luminance from chrominance

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `RgbToHsv(Double)` | Creates a new RGB to HSV conversion. |

## Methods

| Method | Summary |
|:-----|:--------|
| `ApplyAugmentation(ImageTensor<>,AugmentationContext<>)` | Converts the image from RGB to HSV color space. |
| `GetParameters` |  |
| `HsvToRgb(Double,Double,Double)` | Converts HSV values back to RGB. |

